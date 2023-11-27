from multiprocessing.pool import Pool
import dill
import random
import pandas as pd
import pickle as pc
from collections import defaultdict
import networkx as nx

import sys
sys.path.insert(0,'helper/')
import hash


def load_graph_file(file,type2newType,minLimit=5,maxlimit=2000):
    dataset_tmp = pc.load(open(file,'rb'))
    dataset_train={}
    for k, v in dataset_tmp.items():
        if len(v) > minLimit and len(v)<maxlimit:
            for edge, org_type in nx.get_edge_attributes(v,'eventType').items():
                v.edges[edge]['type_edge']=int(type2newType[org_type])
                v.edges[edge].pop('eventType',None)
                
            dataset_train[k]=v#.to_undirected()
    
    print(f'before: {len(dataset_tmp)} after:{len(dataset_train)}')    
    return dataset_train

def load_train_test(mainFile):
    type2newType=pc.load(open('helper/eventTyperInt.pt','rb'))
    dataset_train = load_graph_file(mainFile+'train.pt',type2newType)
    dataset_test = load_graph_file(mainFile+'test.pt',type2newType)
    return dataset_train,dataset_test
    
def findNeignh(graph,start_node,numberOfNeighk):
    neigh = [start_node]
    min_size=min(4,len(graph))
    max_size=min(10,len(graph))
    
    size = random.randint(min_size,max_size)
    # Graph.subgraph neighbor
    initial_graph = graph.subgraph(neigh).copy()
    
    return findNeighPath(graph,initial_graph,numberOfNeighk, start_node, size)

    
def findNeighPath(graph,initial_graph,numberOfNeighk, start_node, size):
    # from initial graph we will continue adding edges
    # until reaching the predefined size
    # global numberOfNeighk
    degrees=graph.degree
    
    visited = set(list(initial_graph.nodes))
    frontiers = [start_node]
    for i in range(random.randint(1,numberOfNeighk)):
        if len(initial_graph)>size:
            return initial_graph
        tmp_front = set(graph.neighbors(frontiers[-1]))
        tmp_front = list(tmp_front - visited)
        if len(tmp_front) == 0:
            break
        new_node = random.choices(tmp_front,weights=(degrees[x]**2 for x in tmp_front))[0]
        frontiers.append(new_node)
        visited.add(new_node)
        edge = (frontiers[-2], frontiers[-1],)
        edge_feature = list(graph.get_edge_data(*edge).items())[0][1]
        initial_graph.add_edge(frontiers[-2], frontiers[-1],type_edge=edge_feature['type_edge'])
    
    g_reverse = graph.reverse()
    frontiers = [start_node]
    for i in range(random.randint(1,numberOfNeighk)):
        if len(initial_graph)>size:
            return initial_graph
        tmp_front = set(g_reverse.neighbors(frontiers[-1]))
        tmp_front = list(tmp_front - visited)
        if len(tmp_front) == 0:
            break
        new_node = random.choices(tmp_front,weights=(degrees[x]**2 for x in tmp_front))[0]
        frontiers.append(new_node)
        visited.add(new_node)
        edge = (frontiers[-1], frontiers[-2],)
        edge_feature = list(graph.get_edge_data(*edge).items())[0][1]
        initial_graph.add_edge(frontiers[-1], frontiers[-2],type_edge=edge_feature['type_edge'])
    

    return initial_graph
        


def find_pos_hashes_mp(nodes_data,numberOfNeighk, graph, node):
    hash.set_nodesdata(nodes_data)
    
    paths = []
    hashSet = set()
    for i in range(2000):
        positiveHash, nodes = find_pos_hash_child_mp(nodes_data,numberOfNeighk, graph, node,hashSet, recursiveCount=0)
        if nodes is None:
            break
        paths.append((positiveHash, nodes))
        hashSet.add(positiveHash)
    return paths

            
def find_pos_hash_child_mp(nodes_data,numberOfNeighk, graph, node,hashSet, recursiveCount=0):
    c=0
    
    if recursiveCount > 40:
        # print('COULD NOT FIND A UNIQUE POS QUERY FOR', node)
        return None, None
    
    nodes = findNeignh(graph,node,numberOfNeighk)
    
    positiveHash = hash.get_hash_targetNodes(nodes, node, numberOfNeighk)
    proc_path = nodes_data.loc[node].path
    # posQueryHashes = defaultdict(lambda: defaultdict(dict))
    if not positiveHash in hashSet:
        return positiveHash, nodes
    else:
        return find_pos_hash_child_mp(nodes_data, numberOfNeighk, graph, node,hashSet, recursiveCount+1)
        

def prepare_run1(t_set,k):
    global THREAD_COUNT
    queryset = []
    for node, graph in t_set.items():
        queryset.append((nodes_data,k, graph, node)) 
    # return queryset

    workers = Pool(THREAD_COUNT)
    records = workers.starmap(find_pos_hashes_mp, queryset)

    print('Worker thread completed')
    for listrecord in records:
        for positiveHash, path in listrecord:
            node = list(path.nodes)[0]
            proc_path = nodes_data.loc[node].path
            posQueryHashes[proc_path][node].add(positiveHash)
            # posQueryHashes[proc_path][positiveHash].add(node)
            hash2graph[node][positiveHash] = path

    # for v in t_set:
    #     print(nodes_data.loc[v].path, len(posQueryHashes[nodes_data.loc[v].path][v]))
        
def createHashes(data_identifier, k):
    global posQueryHashes, hash2graph, datasets
    posQueryHashes = defaultdict(lambda: defaultdict(set))
    hash2graph = defaultdict(dict)
    
    mainFile = f'data/{data_identifier}/k_{k}'
    datasets = load_train_test(mainFile)
    for dataset in datasets:
        prepare_run1(dataset,k)

def createStats(data_identifier, k):
    global posQueryHashStats
    posQueryHashStats = {'train':defaultdict(lambda: defaultdict(int)), 'test':defaultdict(lambda: defaultdict(int))}
    for path, nodes in posQueryHashes.items():
        for start_node, hashes in nodes.items():
            dictKey = 'train' if start_node in datasets[0] else 'test' # if in train
            for hash in hashes:
                posQueryHashStats[dictKey][path][hash]+=1
def createRevDict(data_identifier, k): 
    global hash2seed
    hash2seed = defaultdict(set)
    for path, nodes in posQueryHashes.items():
        for node,hashes in nodes.items():
            for hash in hashes:
                hash2seed[hash].add(node)   

def saveDicts(data_identifier, k):
    save_dir = f'data/{data_identifier}/{k}'
    pc.dump(hash2graph,open(f'{save_dir}hash2graph.pkl','wb'))
    dill.dump(posQueryHashes,open(f'{save_dir}posQueryHashes.pkl','wb'))  
    dill.dump(posQueryHashStats,open(f'{save_dir}posQueryHashStats.pkl','wb')) 
    dill.dump(hash2seed,open(f'{save_dir}hash2seed.pkl','wb'))   

def run(data_identifier,k,thread_count = 20 ):
    global nodes_data, THREAD_COUNT
    THREAD_COUNT=thread_count
    nodes_data=pd.read_csv(f'node_feature/{data_identifier}/all_nodes_data.csv')
    nodes_data=nodes_data.set_index('uuid')
    print(len(nodes_data.type.unique()))
    print(k)
    createHashes(data_identifier, k)
    createStats(data_identifier, k)
    createRevDict(data_identifier, k)                        
    saveDicts(data_identifier, k)

