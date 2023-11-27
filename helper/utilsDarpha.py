#open tar Files
import glob
import networkx as nx
import pandas as pd
import pickle as pc

def openTarFile():
    tarFiles=glob.glob('theia/*.json.tar.gz')
    for tarFile in tarFiles:
        os.system('tar -zxvf '+tarFile)
        fileName=tarFile.split('/')[-1].split('.json')[0]+'.json'
        #os.system('mv '+fileName+' theia/'+fileName)
        os.system('rm '+tarFile)
    os.system('mv *.json* theia/')
    
def createOldInd2newInd(root_file):
    edges_data = pd.read_csv('CSV/'+root_file+'_edges_data.csv')
    type_new=edges_data['Type']
    del edges_data
    edges_data = pd.read_csv('CSV/'+root_file+'_edges_data_old.csv')
    type_old=edges_data['Type']
    del edges_data
    newDf=pd.DataFrame.from_dict({'old':type_old.values, 'new':type_new.values})
    df=newDf.drop_duplicates(subset='old', keep="last")

    oldInd2newInd={}
    for i in range(len(df)):
        row=df.iloc[i]
        oldInd2newInd[row['old']]=row['new']
    return oldInd2newInd

def fixFolder(oldInd2newInd,folder='../neural_subgraph_learning_GNN/data/ta1-theia-e3-official-6r/*.pt'):
    files=glob.glob(folder)
    for file in files:
        fixFile(oldInd2newInd,file)
        

def fixFile(oldInd2newInd,file):
        graphs=pc.load(open(file,'rb'))

        new_graphs={}
        print('createDirectedGraphsmal',len(graphs))
        i=0
        for node,graph in graphs.items():
            print(i, end='\r')
            i+=1
            g=nx.DiGraph()
            for edge, edge_type in nx.get_edge_attributes(graph,'type_edge').items():
                g.add_edge(edge[0],edge[1], type_edge=oldInd2newInd[edge_type])
            new_graphs[node]=g

        pc.dump(new_graphs,open(file,'wb'))
        
def set2dict(setElem):
    return {elem:i for i,elem in enumerate(setElem)}

# def find_types(abstarct_indexer):
#     file_path_types=set()
#     proc_path_types=set()
#     net_types_port=set()
#     net_types_ip=set()
#     for abstract in abstarct_indexer:
#         abstract=abstract.split('_')
#         high_type=abstract[-1]
#         if high_type == 'Proc':
#             proc_path_types.add(''.join(abstract[:-1]))
#         elif high_type == 'File':
#             file_path_types.add(''.join(abstract[:-1]))
#         elif high_type not in ['memory', 'unknown']:
#             net_types_port.add(abstract[0])
#             net_types_port.add(abstract[2])
#             net_types_ip.add(abstract[1])

#     file_path_types=set2dict(file_path_types)
#     proc_path_types=set2dict(proc_path_types)
#     net_types_port=set2dict(net_types_port)
#     net_types_ip=set2dict(net_types_ip)
#     return proc_path_types,file_path_types,net_types_port,net_types_ip

# def abstractindex2array(abstarct_indexer):

#     high_types={'Proc':0,'File':1,'memory':2}
#     proc_path_types,file_path_types,net_types_port,net_types_ip=find_types(abstarct_indexer)

#     len_feature=4+len(proc_path_types)+len(file_path_types)+2*len(net_types_port)+len(net_types_ip)

#     type2array={}
#     for abstract in abstarct_indexer:
#         feat=[0]*len_feature
#         abstract_splited=abstract.split('_')

#         high_type=abstract_splited[-1]
#         if high_type in high_types:
#             feat[high_types[high_type]]=1
#         elif high_type!='unknown':
#             feat[3]=1

#         if high_type == 'Proc':
#             key=''.join(abstract_splited[:-1])
#             key=proc_path_types[key]
#             feat[4+key]=1
            
#         elif high_type == 'File':
#             key=''.join(abstract_splited[:-1])
#             key=file_path_types[key]
#             feat[4+len(proc_path_types)+key]=1
#         elif high_type not in ['memory', 'unknown']:
#             key=net_types_port[abstract_splited[0]]
#             ind=4+len(proc_path_types)+len(file_path_types)
#             feat[ind+key]=1
            
#             ind+=len(net_types_port)
#             key=net_types_ip[abstract_splited[1]]
#             feat[ind+key]=1

#             ind+=len(net_types_ip)
#             key=net_types_port[abstract_splited[2]]
#             feat[ind+key]=1

#         type2array[abstract]=feat
        
#     return type2array

def find_types(abstarct_indexer, isWindows=False):
    file_path_types=set()
    proc_path_types=set()
    net_types_port=set()
    net_types_ip=set()
    reg_types=set()
    for abst in abstarct_indexer:
        abstract=abst.split('_')
        high_type=abstract[-1]
        if high_type == 'Proc':
            proc_path_types.add(''.join(abstract[:-1]))
        elif high_type == 'File':
            file_path_types.add(''.join(abstract[:-1]))
        elif high_type == 'Reg':
            reg_types.add(''.join(abstract[:-1]))
        elif high_type.startswith('/home/') or high_type in ['network0','network53','0','53']:
            file_path_types.add(abst)
        elif high_type not in ['memory', 'unknown']:
#             print(high_type, abst)
            net_types_port.add(abstract[0])
            net_types_port.add(abstract[2])
            net_types_ip.add(abstract[1])

    file_path_types=set2dict(file_path_types)
    proc_path_types=set2dict(proc_path_types)
    net_types_port=set2dict(net_types_port)
    net_types_ip=set2dict(net_types_ip)
    if isWindows:
        reg_types=set2dict(reg_types)
        return proc_path_types,file_path_types,net_types_port,net_types_ip, reg_types
        
    return proc_path_types,file_path_types,net_types_port,net_types_ip, []

def abstractindex2array(abstarct_indexer, isWindows=False):

    high_types={'Proc':0,'File':1,'network':2,'Reg':3}
    proc_path_types,file_path_types,net_types_port,net_types_ip, reg_types=find_types(abstarct_indexer,isWindows=isWindows)
    mainChar = 3
    if isWindows:
        mainChar+=1
        
    len_feature=mainChar+len(proc_path_types)+len(file_path_types)+2*len(net_types_port)+len(net_types_ip)+len(reg_types)

    type2array={}
    for abstract in abstarct_indexer:
        feat=[0]*len_feature
        abstract_splited=abstract.split('_')

        high_type=abstract_splited[-1]
        if high_type == 'Proc':
            feat[0]=1
            key=''.join(abstract_splited[:-1])
            key=proc_path_types[key]
            feat[mainChar+key]=1
            
        elif high_type == 'File':
            feat[1]=1
            key=''.join(abstract_splited[:-1])
            key=file_path_types[key]
            feat[mainChar+len(proc_path_types)+key]=1
            
        elif high_type.startswith('/home/') or high_type in ['network0','network53','0','53']:
            if high_type in ['network0','network53','0','53']:
                feat[1]=2
            else:
                feat[1]=1
            key=file_path_types[abstract]
            feat[mainChar+len(proc_path_types)+key]=1

        elif high_type == 'Reg':
            feat[3]=1
            key=''.join(abstract_splited[:-1])
            key=reg_types[key]
            feat[len_feature-len(reg_types)+key]=1

        else:
            key=net_types_port[abstract_splited[0]]
            ind=mainChar+len(proc_path_types)+len(file_path_types)
            feat[ind+key]=1
            
            ind+=len(net_types_port)
            key=net_types_ip[abstract_splited[1]]
            feat[ind+key]=1

            ind+=len(net_types_ip)
            key=net_types_port[abstract_splited[2]]
            feat[ind+key]=1

        type2array[abstract]=feat
        
    return type2array


def abstractArrayCheck(abstarct_indexer,type2array):

    proc_path_types,file_path_types,net_types_port,net_types_ip=find_types(abstarct_indexer)
    clms=['Proc','File','memory','network']
    clms+=[elem+'_proc' for elem in proc_path_types]
    clms+=[elem for elems in [file_path_types,net_types_port,net_types_ip] for elem in elems]
    clms+=[elem+'_dst' for elem in net_types_port]
    dic4df={}
    dic4df['abstract']=list(type2array.keys())
    for i,clm in enumerate(clms):
        dic4df[clm]=[vl[i] for vl in type2array.values()]

    pd.set_option('display.max_columns', None)   
    return pd.DataFrame.from_dict(dic4df)

def findProcFeature(path):
    file2feature={}
    files=glob.glob(path)
    for file in files:
        proc=file.split('/')[-1][:-3]
        file2feature[proc]=[]
        chekFun=False
        for line in open(file):
            if 'functions:' in line:
                chekFun=True
            if not chekFun:
                continue
            if line[:2]=='  ' and line[3]!=' ':
                file2feature[proc].append(line.replace(' ','').split(':')[0])
                
    file2feature['python2.7']=file2feature['python']
    file2feature['sudo']=file2feature['su']
    file2feature['firefox']=['browser']
    file2feature['thunderbird']=['browser']
    return file2feature
        