import argparse
import time

import numpy as np
import torch

from common import utils


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({'font.size': 16})
import seaborn as sns

import random
from sklearn.manifold import TSNE
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

#it takes time due to added line
from common import data
####
from common import models
from subgraph_matching.config import parse_encoder

from subgraph_matching import train
import importlib as imp
import pickle as pc
import argparse

from subgraph_matching import config 
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    config.parse_encoder(parser)
    args = parser.parse_args()
    
    sampling_stats = f'data/{args.data_identifier}/test_neg_dict_{args.numberOfNeighK}.pc'
    if not os.path.exists(sampling_stats):
        print('first sampling stats will be created')
        print('this is a one time process for each dataset')
        import create_neg
        import create_pos_neg_dict as cd
        create_neg.run(args.numberOfNeighK,args.data_identifier)
        cd.run(args.data_identifier,args.numberOfNeighK) 
      
    ### load dataset and set utils parameters accordingly
    utils.loadDatas(feature=args.data_identifier, numberOfNeighK=args.numberOfNeighK, args=args)
    utils.numberOfFeature = len(list(utils.abstractType2array.values())[0])
    args.feature_size = utils.numberOfFeature + 17
    utils.glob_feature=args.data_identifier
    
    train.train_loop(args)