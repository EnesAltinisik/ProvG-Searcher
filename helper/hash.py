from hashlib import sha256

nodes_data = None
unknowns = set()

def set_nodesdata(data):
    global nodes_data
    nodes_data = data

def getHash(input_):
    return sha256(input_.encode('utf-8')).hexdigest()

# def get_abstract(node):
#     assert nodes_data is not None
#     return str(nodes_data.loc[node.split('_')[0]].type)

def get_abstract(node):
    try:
        return str(nodes_data.loc[node.split('_')[0]].type)
    except:
        if node.split('_')[0] not in unknowns:
            unknowns.add(node.split('_')[0])
#             print('unknown node:',node.split('_')[0])
        return 'unknown_File'

def getNeighStruct(G,node,k,key,nodeHash):
    
    retSet=set([nodeHash[node][k-1][key]]) #start with it shelf
    c=0
    for (nd,neigh,data) in G.edges(node,data=True):
        c+=1
        retSet.add(str(data['type_edge'])+nodeHash[neigh][k-1][key])
        
    if c==0:
        return nodeHash[node][k-1][key]
    
    retStr = ';'.join(retSet)
    return getHash(retStr)

def get_hash_targetNodes(g_t, seed_node, K=4):
    nodeHash={}
    g_t_rev=g_t.reverse()
    
    for node in g_t.nodes:
        nodeHash[node]={}
        nodeHash[node][0]={}
        abstNode=get_abstract(node)
        nodeHash[node][0]['back']=abstNode
        nodeHash[node][0]['forw']=abstNode
    
    for k in range(1,K):
        for node in g_t.nodes:
            nodeHash[node][k]={}
            nodeHash[node][k]['back'] = getNeighStruct(g_t_rev,node,k,'back',nodeHash)
            nodeHash[node][k]['forw'] = getNeighStruct(g_t,node,k,'forw',nodeHash)

    nodeHash[seed_node][K]={}
    nodeHash[seed_node][K]['back'] = getNeighStruct(g_t_rev,seed_node,K-1,'back',nodeHash)
    nodeHash[seed_node][K]['forw'] = getNeighStruct(g_t,seed_node,K-1,'forw',nodeHash)

    hashValue = getHash(nodeHash[seed_node][K]['back'] + nodeHash[seed_node][K]['forw'])

    return hashValue
