import pickle as pc
import networkx as nx
import pandas as pd
import numpy as np

def run(numberOfLAyerK,dataIndentifier):

    for testTrain in ['test','train']:

        dataset_train = pc.load(open(f'data/{dataIndentifier}/k_{numberOfLAyerK}{testTrain}.pt','rb'))
        dataset_tmp={}
        for k, v in dataset_train.items():
            if len(v) > 5 and len(v)<2000:
                dataset_tmp[k]=v


        print(len(dataset_train),len(dataset_tmp))
        del dataset_train  

        nodes_data  = pd.read_csv(f'node_feature/{dataIndentifier}/nodes_data.csv')

        NFQ=40
        save_file=f'data/{dataIndentifier}/{testTrain}_neg_dict_{numberOfLAyerK}.pc'

        proc_type=[tp for tp in set(nodes_data.type) if '_Proc' in tp]

        def get_neigh(g,seed_proc,tip):
            ins=set([edge[0] for edge in g.in_edges(seed_proc)])
            ins=ins.intersection(proc_uuids)
            outs=set([edge[1] for edge in g.out_edges(seed_proc)])
            outs=outs.intersection(proc_uuids)
            df=nodes_data[nodes_data['uuid'].isin(list(ins.union(outs)))]
            return df,set(df[tip].values)

        def get_neigh_with_path(df,paths):
            return set(df[df['path'].isin(paths)]['uuid'].values)

        df=nodes_data[nodes_data['uuid'].isin(list(dataset_tmp.keys()))]
        proc_uuids=set(df['uuid'].values)

        prc_df=df[df['type'].isin(proc_type)]
        uniq_paths=list(set(prc_df.path))
        proc_diffs={}

        for proc in uniq_paths: 
            if type(proc)==float:
                continue
            proc_diffs[proc]={}
            same_procs=prc_df[prc_df['path']==proc]['uuid'].values

            neighs={seed_node:get_neigh(dataset_tmp[seed_node],seed_node,'path') for seed_node in same_procs}
            for i, seed_node in enumerate(same_procs):
                neigh=neighs[seed_node][1]
                proc_diffs[proc][seed_node]={}

                for j in range(len(same_procs)):
                    if i==j:
                        continue
                    seed_node_j=same_procs[j]
                    neigh_j=neighs[seed_node_j][1]
                    diffs=neigh_j.difference(neigh)

                    if len(diffs)>0:
                        proc_diffs[proc][seed_node][seed_node_j]=get_neigh_with_path(neighs[seed_node_j][0],diffs)

        samePathCandidates={}
        for path, procs in proc_diffs.items():
            for proc, candidateProcs in procs.items():
                for candidateProc, candidateNodes in candidateProcs.items():
                    if proc not in samePathCandidates:
                        samePathCandidates[proc]={}
                    samePathCandidates[proc][candidateProc]=candidateNodes


        nodes_data_proc=nodes_data[nodes_data['uuid'].isin(proc_uuids)]
        len(nodes_data_proc)

        sameAbsCandidates={}
        pathCand={}
        for proc in dataset_tmp:
            proc_info=nodes_data_proc[nodes_data_proc['uuid']==proc].iloc[0]
            path=proc_info.path
            if path in pathCand:
                abs_proc=pathCand[path]
            else:
                abs_proc=nodes_data_proc[nodes_data_proc['type']==proc_info.type]
                abs_proc=abs_proc[abs_proc['path']!=proc_info.path].uuid.values
                pathCand[path]=abs_proc

            if len(abs_proc)>0:
                sameAbsCandidates[proc]=abs_proc

        len(sameAbsCandidates)   

        import math
        from collections import defaultdict
        import random

        # combine and create count
        procCandidates={}
        revDict=defaultdict(dict)
        for proc in proc_uuids:
            procCandidates[proc]={'other':{},'normal':{}}

            if proc in sameAbsCandidates and proc in samePathCandidates:
                baseValues=NFQ/2
            else:
                baseValues=2*NFQ/3

            if proc in sameAbsCandidates:
                vl=math.ceil(baseValues/len(sameAbsCandidates[proc]))
                possibleProcs=[]
                for cnd in sameAbsCandidates[proc]:
                    possibleProcs.extend([cnd]*vl)
                possibleProcs=random.sample(possibleProcs,int(baseValues))

                for cnd in set(possibleProcs):
                    vl=possibleProcs.count(cnd)
                    procCandidates[proc]['other'][cnd]=vl
                    revDict[cnd][proc]=vl


            if proc in samePathCandidates:
                vl=math.ceil(baseValues/len(samePathCandidates[proc]))
                possibleProcs=[]
                for cnd in samePathCandidates[proc]:
                    possibleProcs.extend([cnd]*vl)
                possibleProcs=random.sample(possibleProcs,int(baseValues))

                for cnd in set(possibleProcs):
                    vl=possibleProcs.count(cnd)
                    procCandidates[proc]['normal'][cnd]={'val':vl,'cndNod':samePathCandidates[proc][cnd]}
                    revDict[cnd][proc]=vl


        #use reverse dict for eliminate
        for proc in proc_uuids:
            getMultiples=[]
            for pr, vl in revDict[proc].items():
                getMultiples.extend([pr]*vl)
            if len(getMultiples)<2*NFQ:
                continue

            getMultiplesNew=random.sample(getMultiples,2*NFQ)
            for pr in set(getMultiples):
                cnt=getMultiplesNew.count(pr)
                if cnt==0:
                    revDict[proc].pop(pr,None)
                    if proc in procCandidates[pr]['other']:
                        procCandidates[pr]['other'].pop(proc,None)
                    else:
                        procCandidates[pr]['normal'].pop(proc,None)

                else:
                    revDict[proc][pr]=cnt
                    if proc in procCandidates[pr]['other']:
                        procCandidates[pr]['other'][proc]=cnt
                    else:
                        procCandidates[pr]['normal'][proc]['val']=getMultiples.count(pr)  
                        
        paths=[]
        other_tot=[]
        other_proc=[]
        same_tot=[]
        same_proc=[]
        rand_lmt=[]
        for proc in procCandidates:
            path=nodes_data_proc[nodes_data_proc['uuid']==proc].iloc[0].path
            paths.append(path)

            other_tot.append(sum(list(procCandidates[proc]['other'].values())))
            other_proc.append(len(procCandidates[proc]['other']))

            same_tot.append(sum([vl['val'] for vl in procCandidates[proc]['normal'].values()]))
            same_proc.append(len(procCandidates[proc]['normal']))

            rnd_lmt=max(0,(NFQ/5)*4-other_tot[-1]-same_tot[-1])
            rand_lmt.append(rnd_lmt)
            procCandidates[proc]['rand']=rnd_lmt
    

        pc.dump(procCandidates,open(save_file,'wb'))

