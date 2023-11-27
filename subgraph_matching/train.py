"""Train the order embedding model"""

# Set this flag to True to use hyperparameter optimization
# We use Testtube for hyperparameter tuning
HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

from common import data
from common import models
from common import utils
if HYPERPARAM_SEARCH:
    from test_tube import HyperOptArgumentParser
    from subgraph_matching.hyp_search import parse_encoder
else:
    from subgraph_matching.config import parse_encoder
from subgraph_matching.test import validation

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch

import pickle as pc
def build_model(args):
    # build model
    if args.method_type == "order":
        model = models.OrderEmbedder(args.feature_size, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(args.feature_size, args.hidden_dim, args)
    model.to(utils.get_device())
    if os.path.exists(args.model_path):
        print('Model is loaded !!!!!!!!!')
        model.load_state_dict(torch.load(args.model_path,
            map_location=utils.get_device()))
    
    
    return model

def make_data_source(args,isTrain=True):
    
    dirtyGraph=None
    data_source = data.DiskDataSource(args.dataset, args.data_identifier,args.numberOfNeighK,#6,26,
                node_anchored=args.node_anchored, feature_type=args.feature_type)
    
    return data_source

def train(args, model,data_source, scheduler, opt,clf_opt, batch_n):
    """Train the order embedding model.

    args: Commandline arguments
    logger: logger for logging progress
    in_queue: input queue to an intersection computation worker
    out_queue: output queue to an intersection computation worker
    """
    loaders = data_source.gen_data_loaders(args.batch_size, args.batch_size, train=True)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        model.emb_model.train()
        model.emb_model.zero_grad()
        
        
        batch = data_source.gen_batch(batch_target, batch_neg_target, batch_neg_query, True)
        batch = [elem.to(utils.get_device()) for elem in batch]
        pos_a, pos_b, neg_a, neg_b = batch
        
        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)
        
        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(
            utils.get_device())
        
        pred = model(emb_as, emb_bs)
        loss = model.criterion(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        if args.method_type == "order":
            model.emb_model.eval()
            with torch.no_grad():
                pred = model.predict(pred)
            model.clf_model.zero_grad()
            pred = model.clf_model(torch.nn.functional.sigmoid(pred.unsqueeze(1)))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()
        
        pred = pred.argmax(dim=-1)
        acc = torch.mean((pred == labels).type(torch.float))
        
        train_loss = loss.item()
        train_acc = acc.item()

        yield train_loss, train_acc
        

def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")


    print("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)

    model = build_model(args)

    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=0.01)
    else:
        clf_opt = None

    print('create test points')
    batch_n = 0
    data_source = make_data_source(args,isTrain=False)
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
        train=False, use_distributed_sampling=False)
    
    test_pts = []
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
        batch_n += 1

    if args.test:
        val_stats=validation(args, model, test_pts, logger, 0, 0, verbose=True)
        return val_stats
    
    print('test points are created')
    print('load only train data')
    data_source = make_data_source(args)  
        
    scheduler, opt = utils.build_optimizer(args, model.emb_model.parameters())
        
    train_stats = []
    val_stats = []
    batch_n = 0
    for epoch in range(args.n_batches):
        for train_loss, train_acc in train(args, model,data_source,scheduler, opt,clf_opt, batch_n):
            print("Batch {}. Loss: {:.4f}. Training acc: {:.4f}".format(
                batch_n, train_loss, train_acc), end="               \r")
            train_stats.append((train_loss, train_acc))
            logger.add_scalar("Loss/train", train_loss, batch_n)
            logger.add_scalar("Accuracy/train", train_acc, batch_n)
            batch_n += 1
        
        if epoch %  args.eval_interval == 0:
            val_stat = validation(args, model, test_pts, logger, batch_n, epoch,verbose=True)

def main(force_test=False):
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        train_loop(args)

if __name__ == '__main__':
    main()
