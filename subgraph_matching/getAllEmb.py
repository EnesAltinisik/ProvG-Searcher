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

import torch.multiprocessing as mp

try:
     mp.set_start_method('spawn')
     mp.set_sharing_strategy('file_system')
    
except RuntimeError:
    pass


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

def make_data_source(args):
    toks = args.dataset.split("-")
    if toks[0] == "syn":
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = data.OTFSynDataSource(
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = data.OTFSynImbalancedDataSource(
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    else:
        if 'darpha' in args.dataset or len(toks) == 1 or toks[1] == "balanced":
            data_source = data.DiskDataSource(args.dataset, args.data_identifier,
                node_anchored=args.node_anchored, feature_type=args.feature_type)
        elif toks[1] == "imbalanced":
            data_source = data.DiskImbalancedDataSource(toks[0],
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    return data_source



def getemb(args):
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
#     model = nn.DataParallel(model,device_ids = [1,4,5]).module
#     model.to("cuda")
    model.share_memory()

    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
    else:
        clf_opt = None

    model.eval()
    
    data_source = make_data_source(args)
    
    procEmb={}
    
    while True:
        procs, graphs = data_source.get_all_graph(512)
        if len(procs)==0:
            return procEmb
        with torch.no_grad():
            emb_graphs = model.emb_model(graphs)

            for i, proc in enumerate(procs):
                procEmb[proc]=emb_graphs[i]
            
        
        
def main(force_test=False):
    mp.set_start_method("spawn", force=True)
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    # Currently due to parallelism in multi-gpu training, this code performs
    # sequential hyperparameter tuning.
    # All gpus are used for every run of training in hyperparameter search.
    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        train_loop(args)

if __name__ == '__main__':
    main()
