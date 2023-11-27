"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from common import utils
from common import feature_preprocess
import pickle as pc

# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred
#         diff_emb = emb_bs - emb_as
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e
    
    def criterion(self, pred, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0); 
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e_org = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=utils.get_device()), emb_bs - emb_as)**2, dim=1)

        e_org[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), self.margin - e_org)[labels == 0]
        
        return torch.sum(e_org)


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            print(dsfjkjkdfjk)
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                if args.conv_type == "SAGE_typed":
                    self.convs.append(conv_model(SAGEConv, hidden_input_dim, hidden_dim,n_edge_types=args.n_edge_type))
                elif args.conv_type == "GCN_typed":
                    self.convs.append(conv_model(pyg_nn.GCNConv, hidden_input_dim, hidden_dim,n_edge_types=args.n_edge_type))
                elif args.conv_type == "GIN_typed":
                    self.convs.append(conv_model(GINConv, hidden_input_dim, hidden_dim,n_edge_types=args.n_edge_type))
                else:
                    self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3
            
            
        self.pool=args.pool
        
        if self.pool == 'set':
            self.poolingLayer = pyg_nn.GraphMultisetTransformer(post_input_dim, args.batch_size, hidden_dim)
        else:  
            self.post_mp = nn.Sequential(
                nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 256), nn.ReLU(),
                nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type
        
           

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            #return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(i, h)
        elif 'typed' in model_type:
            print(model_type)
            return ConvEdgeType
        
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        x, edge_index, batch, edge_type = data.node_feature, data.edge_index, data.batch, data.type_edge
        indsOfSeeds = torch.where(x[:,0]==1)[0]
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    if 'typed' in self.conv_type:
                        x = self.convs[i](curr_emb, edge_index, edge_type)
                    else:
                        x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                if 'typed' in self.conv_type:
                    x = self.convs[i](curr_emb, edge_index, edge_type)
                else:
                    x = self.convs[i](x, edge_index)
                    
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        if self.pool == 'only_seed':
            emb = emb[indsOfSeeds]
            emb = self.post_mp(emb)
        elif self.pool == 'add':
            emb = pyg_nn.global_add_pool(emb, batch)
            emb = self.post_mp(emb)
        elif self.pool == 'mean':
            emb = pyg_nn.global_mean_pool(emb, batch)
            emb = self.post_mp(emb)
        elif self.pool == 'set':
            emb = self.poolingLayer(emb, batch ,edge_index=edge_index)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)    

    
class ConvEdgeType(nn.Module):
    def __init__(self, conv_type, in_channels, out_channels, n_edge_types=1):
        super(ConvEdgeType, self).__init__()
        
        # pyg_nn.GCNConv, SAGEConv, GINConv
        
        self.modelAs = nn.ModuleList(
            [conv_type(in_channels, out_channels,aggr="add") for _ in range(n_edge_types)]
        )
        self.modelBs = nn.ModuleList(
            [conv_type(in_channels, out_channels,aggr="add",flow='target_to_source') for _ in range(n_edge_types)]
        )
            
        self.lin = nn.Linear(2*out_channels*n_edge_types, out_channels)
        self.out_channels=out_channels
     
    def forward(self, x, edge_index, edge_type=None,edge_weight=None, size=None,
                res_n_id=None):
        
        edge_index, edge_type = pyg_utils.remove_self_loops(edge_index,edge_attr=edge_type)
        
        
        for i in range(len(self.modelAs)):
            
            if edge_type==None:
                edge_index_type = edge_index
            else:
                eids = torch.nonzero(edge_type == i, as_tuple=False).view(-1)
                srcs=edge_index[0][eids]
                dsts=edge_index[1][eids]
                edge_index_type = torch.stack((srcs,dsts),0)

            x1 = self.modelAs[i](x, edge_index_type)
            x2 = self.modelBs[i](x, edge_index_type)
            
            if i==0:
                emb = torch.cat((x1, x2), dim=1)
            else:
                emb = torch.cat((emb,x1, x2), dim=1)
            
        return self.lin(emb)
       
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    
class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add",flow='source_to_target'):
        super(SAGEConv, self).__init__(aggr=aggr,flow=flow)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        #edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        try:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
            return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)
        except:
            print(edge_index)
            pc.dump([edge_index, size, x, edge_weight, res_n_id],open('edge_index.pkl','wb'))
            dfghjk

    def message(self, x_j, edge_weight):
        #return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        
        self.nn = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

