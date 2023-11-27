import os
import pickle
import time
import random

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset, Generator
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.multiprocessing as mp
from multiprocessing.pool import Pool
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
import scipy.stats as stats

from common import feature_preprocess
from common import utils
import gc
gc.enable()

import pickle as pc
import networkx as nx

import pickle as pc
type2newType=pc.load(open('helper/eventTyperInt.pt','rb'))
import random

def load_train(file,feature_type,minLimit=5,maxlimit=2000):
    dataset_tmp = pc.load(open(file,'rb'))
    dataset_train={}
    for k, v in dataset_tmp.items():
        if len(v) > minLimit and len(v)<maxlimit:
            for edge, org_type in nx.get_edge_attributes(v,'eventType').items():
                try:
                    v.edges[edge]['type_edge']=int(type2newType[org_type])
                except:
                    v.edges[edge]['type_edge']=len(type2newType)
                v.edges[edge].pop('eventType',None)
                
            dataset_train[k.split('_')[0]]=v#.to_undirected()
    
    print(f'before: {len(dataset_tmp)} after:{len(dataset_train)}')    
    
    return dataset_train
    
def load_dataset(name,feature_type):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    file=name.replace('.pt','train.pt')
    dataset_train=load_train(file,feature_type)
    file=name.replace('.pt','test.pt')
    dataset_test=load_train(file,feature_type)
    
    return dataset_train, dataset_test, "graph"

class DataSource:
    def gen_batch_child(batch_target, batch_neg_target, batch_neg_query, train):
        print('errorrrrrr!!!!!!')
        raise NotImplementedError
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

def removekey(d, key):
    r = d[key]
    del d[key]
    return r

def accepts(g, path):
    return all([(path[i],path[i+1]) in g.edges() for i in range(len(path)-1)])

def rename(G,nd_datas):
    node_list=list(G.nodes)
    node_type=nd_datas.loc[node_list].type_index
    rename=dict(zip(node_list,node_type))
    return nx.relabel_nodes(G, rename)

def isSub(tg,qg,nd_datas):
    tg=rename(tg,nd_datas)
    qg=rename(qg,nd_datas) 
    qg_nodes=list(qg.nodes)
    for node in qg_nodes:
        for nd in qg.neighbors(node):
            for nd2 in qg.neighbors(nd):
                for nd3 in qg.neighbors(nd2):
                    if not accepts(tg, [node,nd,nd2,nd3]):
                        return True
                if not accepts(tg, [node,nd,nd2]):
                    return True
            if not accepts(tg, [node,nd]):
                return True
        if not node in tg.nodes:
            return True
    return False

def isSubPar(g_q,g_t,nd_datas):
    typ,cnt=np.unique(nd_datas.loc[list(g_q.nodes)].type.values,return_counts=True)
    sourceDict=dict(zip(typ,cnt))
    typ,cnt=np.unique(nd_datas.loc[list(g_t.nodes)].type.values,return_counts=True)
    dstDict=dict(zip(typ,cnt))
    diff=0
    for typ,cnt in sourceDict.items():
        if typ in dstDict:
            diff+=max(0,cnt-dstDict[typ])
        else:
            diff+=cnt
            
    if diff==0:
        return isSub(g_t.copy(),g_q.copy(),nd_datas)
        
    return True

def createSub(q_a,q_a_rev,a_a,i=2):
    neigh=set([a_a])
    for ng in q_a.neighbors(a_a):
        neigh.add(ng)
        if i==2:
            for ng1 in q_a.neighbors(ng):
                neigh.add(ng1)

    for ng in q_a_rev.neighbors(a_a):
        neigh.add(ng)
        if i==2:
            for ng1 in q_a_rev.neighbors(ng):
                neigh.add(ng1)
    return neigh    

def isNotSubGraph(g_q,g_t,seed_q,seed_t,nd_datas,isTrain=True):
    if not isTrain:
        # be more accurate for test set
        q_edges = getEdges(g_q,nd_datas)
        t_edges = getEdges(g_t,nd_datas)
        if len(q_edges-t_edges)==0:
            return False
        
    if len(g_t)<500:
        if isSubPar(g_q,g_t,nd_datas):
            return True
    g_q_rev=g_q.reverse()
    g_t_rev=g_t.reverse()
    for i in range(1,3):
        neigh_q=createSub(g_q,g_q_rev,seed_q,i=i)
        neigh_t=createSub(g_t,g_t_rev,seed_t,i=i)
        if isSubPar(g_q.subgraph(neigh_q),g_t.subgraph(neigh_t),nd_datas):
            return True
    return False

def getEdges(q_g,nodes_data):
    query_nodes = nodes_data.loc[list(q_g.nodes)]
    df = nx.to_pandas_edgelist(q_g)
    df = df.merge(query_nodes,left_on='source',right_index=True,how='left')
    df = df.merge(query_nodes,left_on='target',right_index=True,how='left')
    df=df[['type_edge','type_x','type_y']]
#     df=df[['type_edge','path_x','path_y']]
    df['type_edge']=df['type_edge'].apply(str)
    df['type_x']=df['type_x'].apply(str)
    df['type_y']=df['type_y'].apply(str)
    edges = df.apply(lambda x: ';'.join(x.values.tolist()),axis=1)
    edges=set(edges)
    return edges

class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, data_identifier, numberOfNeighK, node_anchored=False, min_size=10,
        max_size=20, feature_type='type', dirtyGraph=None):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name,feature_type)
        self.min_size = min_size
        self.max_size = max_size
        self.feature_type = feature_type
        self.data_identifier=data_identifier
        self.get_all_ind=0
        self.combined=False
        self.initialazeArrays()
        self.numberOfNeighK=numberOfNeighK        
        
       
    def initialazeArrays(self):
        self.pos_a=[]
        self.pos_a_anchors=[]
        self.pos_b=[]
        self.pos_b_anchors=[]
        self.neg_a=[]
        self.neg_a_anchors=[]
        self.neg_b=[]
        self.neg_b_anchors=[]

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders
        
    def record2self(self,record):
        pos_a,pos_a_anchors,pos_b,pos_b_anchors,neg_a,neg_a_anchors,neg_b,neg_b_anchors = record
        pos_a = utils.feature_graphs(pos_a, self.feature_type, self.data_identifier, numberOfNeighK=self.numberOfNeighK, anchors=pos_a_anchors)
        pos_b = utils.feature_graphs(pos_b, self.feature_type, self.data_identifier, numberOfNeighK=self.numberOfNeighK, anchors=pos_b_anchors)
        neg_a = utils.feature_graphs(neg_a, self.feature_type, self.data_identifier, numberOfNeighK=self.numberOfNeighK, anchors=neg_a_anchors)
        neg_b = utils.feature_graphs(neg_b, self.feature_type, self.data_identifier, numberOfNeighK=self.numberOfNeighK, anchors=neg_b_anchors)

        self.pos_a.extend(pos_a)
        self.pos_a_anchors.extend(pos_a_anchors)
        self.pos_b.extend(pos_b)
        self.pos_b_anchors.extend(pos_b_anchors)
        self.neg_a.extend(neg_a)
        self.neg_a_anchors.extend(neg_a_anchors)
        self.neg_b.extend(neg_b)
        self.neg_b_anchors.extend(neg_b_anchors)
        
    def gen_batch_child(self, a, b, c, train, max_size=25, min_size=15, seed=None,
        filter_negs=False, sample_method="tree-pair"):
        record = gen_batch_childMP(a,b, c, train,self.node_anchored,self.dataset, max_size, min_size, seed, filter_negs, sample_method)
        self.record2self(record)
    
    
    def get_all_graph(self, batch_size):
        if not self.combined:
            train_set, test_set, task = self.dataset
            print(len(train_set),len(test_set))
            self.dataset=train_set
            self.dataset.update(test_set)
            print(len(self.dataset))
            self.combined=True
            
        if len(self.dataset) < 3:
            return [], None
            
        graphs = []
        procs = []
        for i, key in enumerate(self.dataset.keys()):
            if i==batch_size:
                break
                
            graph = self.dataset[key]
            graphs.append(graph)
            procs.append(key)
        
        for key in procs:
            del self.dataset[key]
            
        graphs = utils.feature_graphs(graphs, self.feature_type, self.data_identifier, anchors=procs)
        
        graphs = utils.batch_nx_graphs(graphs, anchors=procs, feature_type=self.feature_type)
        return procs, graphs
                
        
    def gen_batch(self, a, b, c, train,max_size=25, min_size=15, seed=None,filter_negs=False, sample_method="tree-pair"):
        
        batch_size = a
        self.initialazeArrays()
        self.gen_batch_child(a, b, c, train, max_size=max_size, min_size=min_size, seed=seed,
                            filter_negs=filter_negs, sample_method=sample_method)
            
    
        pos_a = utils.batch_nx_graphs(self.pos_a, anchors=self.pos_a_anchors if
            self.node_anchored else None,feature_type=self.feature_type)
        pos_b = utils.batch_nx_graphs(self.pos_b, anchors=self.pos_b_anchors if
            self.node_anchored else None,feature_type=self.feature_type)
        neg_a = utils.batch_nx_graphs(self.neg_a, anchors=self.neg_a_anchors if
            self.node_anchored else None,feature_type=self.feature_type)
        neg_b = utils.batch_nx_graphs(self.neg_b, anchors=self.neg_b_anchors if
            self.node_anchored else None,feature_type=self.feature_type)

        return pos_a, pos_b, neg_a, neg_b
    
def gen_batch_childMP(a, b, c, train,node_anchored,dataset, max_size=25, min_size=15, seed=None,
    filter_negs=False, sample_method="tree-pair"):

    min_size=2
    max_size=15
    batch_size = a
    train_set, test_set, task = dataset
#         global dataset
    graphs = train_set if train else test_set
    if seed is not None:
        random.seed(seed)

    pos_a, pos_b = [], []
    pos_a_anchors, pos_b_anchors = [], []

    while len(pos_a) < (batch_size // 2):  

        size = random.randint(min_size+1, max_size)
        graph, a = utils.get_graph_nodes(graphs)
        _,b = utils.create_pos_query(graph, a[0], train)

        if b is None:
            continue

        if node_anchored:
            anchor = a[0]
            pos_a_anchors.append(anchor)
            pos_b_anchors.append(anchor)
#             neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)#.copy()
        neigh_a, neigh_b = graph, b 

        pos_a.append(neigh_a)
        pos_b.append(neigh_b)

    neg_a, neg_b = [], []
    neg_a_anchors, neg_b_anchors = [], []
    while len(neg_a) < (batch_size // 2):

        size = random.randint(min_size+1, max_size)
        graph_a, a = utils.get_graph_nodes(graphs)
        graph_b, b, neg_source, b_anchor = utils.create_neg_query(graphs,a[0],train)

        if b is None:
            continue

        neigh_a, neigh_b = graph_a, b

        if not isNotSubGraph(neigh_b,neigh_a,b_anchor,a[0],utils.nodes_data,isTrain=train):
            continue

        if node_anchored:
            neg_a_anchors.append(a[0])
            neg_b_anchors.append(b_anchor)
        
        neg_a.append(neigh_a)
        neg_b.append(neigh_b)

    return pos_a,pos_a_anchors,pos_b,pos_b_anchors,neg_a,neg_a_anchors,neg_b,neg_b_anchors
        
