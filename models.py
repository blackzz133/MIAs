import argparse
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import torch
from torch_geometric_temporal.nn.recurrent import DCRNN, EvolveGCNO, TGCN, A3TGCN, GConvGRU
import pytorch_lightning as pl
#from torch.nn import Dropout
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch.nn as nn
from ModelExtraction.StaticGraphTemporalSignal.config import *
from layers import StructuralAttentionLayer, TemporalAttentionLayer,TemporalAttentionLayer2
import torch_geometric
#from dataset_loader import DBLPELoader
from TGCN.signal import temporal_signal_split
#from ModelExtraction.active_learning.utils import *
#from ModelExtraction.active_learning import *
import random
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch.nn import LSTM
#from config import *
from torch_scatter import scatter

class Robust_RGNN(torch.nn.Module):
    def __init__(self, args, device, num_features, time_length, num_classes, model_type):
        super(Robust_RGNN, self).__init__()
        self.args = args
        self.num_time_steps = time_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.structural_attn1, self.recurrent_layer1,self.temporal_attn, self.linear = self.build_model(num_features, num_classes, model_type)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.temporal_layer_config[0]),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.temporal_layer_config[0] * 2, 2),
        )
        self.output = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)
        #, self.temporal_attn
    def build_model(self, num_features, num_classes, model_type):
        input_dim = self.num_features
        structural_attention_layers = torch.nn.Sequential()
        # num of structure layer is two
        structural_attn1 = StructuralAttentionLayer(input_dim=input_dim,
                                                        output_dim=self.structural_layer_config[0],
                                                        n_heads=8,
                                                        attn_drop=self.spatial_drop,
                                                        ffd_drop=self.spatial_drop,
                                                        residual=self.args.residual)
        if model_type == 'DCRNN':
            recurrent_layer1 = DCRNN(self.structural_layer_config[0],
                                     self.temporal_layer_config[0], 1)
        elif model_type == 'GConvGRU':
            recurrent_layer1 = GConvGRU(self.structural_layer_config[0],
                                     self.temporal_layer_config[0], 1)
        elif model_type == 'TGCN':
            recurrent_layer1 = TGCN(self.structural_layer_config[0],
                                     self.temporal_layer_config[0], 1)
        elif model_type == 'A3TGCN':
            recurrent_layer1 = A3TGCN(self.structural_layer_config[0],
                                     self.temporal_layer_config[0], 1)

        #temporal_attn = torch.nn.Linear(in_features=self.temporal_layer_config[0], out_features=self.num_classes)
        temporal_attn = TemporalAttentionLayer2(input_dim=self.temporal_layer_config[0],
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
        linear = torch.nn.Linear(in_features=self.temporal_layer_config[0], out_features=self.num_classes)

        return structural_attn1, recurrent_layer1, temporal_attn, linear

    def forward(self, graphs):
        st_out = None
        coe1 = []
        hidden = None
        '''
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out2, c2 = self.structural_attn2(out1)
            out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
            st_out.append(self.temporal_attn(out3))
            coe1.append(c1)
            coe2.append(c2)
            #st_out.append(out3)
        '''
        if type(graphs) != list:
            graphs = data_preprossing(graphs)

        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out1.x = out1.x.relu()
            try:
                out2, hidden = self.recurrent_layer1(out1.x, out1.edge_index, out1.edge_attr, hidden)
            except:
                try:
                    out2 = self.recurrent_layer1(out1.x, out1.edge_index, out1.edge_attr, hidden)
                except:
                    out2 = self.recurrent_layer1(out1.x[:,:,None], out1.edge_index, out1.edge_attr, hidden)


            #st_out.append(self.temporal_attn(out3))
            coe1.append(c1)
            #if self.training:
            #out2 = self.drop(out2)
            #out2 = out2.relu()
            #y = self.labels(graph.y)
            #out3 = self.combine(torch.cat((out2, y), 1))
            if st_out == None:
                st_out = out2[None, :, :].transpose(0,1)
            else:
                st_out = torch.cat((st_out,out2[None, :, :].transpose(0,1)), dim=1)
            #st_out:[T,N,F]
        temporal_out = self.temporal_attn(st_out)
        temporal_out = temporal_out.relu()
        #return st_out, coe1, coe2
        return self.linear(temporal_out.transpose(0,1)), coe1
        #return self.linear(st_out.transpose(0, 1)), coe1, coe2
    '''
    def forward2(self, graphs):
        st_out = []
        coe1 = []
        coe2 = []
        hidden = None
        for t in range(len(graphs)):
            graph = graphs[t].cuda()
            out1, c1 = self.structural_attn1(graph)
            out2, c2 = self.structural_attn2(out1)
            out3, hidden = self.recurrent_layer1(out2.x[None, :, :], hidden)
            st_out.append(self.temporal_attn(out3).squeeze())
            coe1.append(c1)
            coe2.append(c2)
            #st_out.append(out3)
        #temporal_out = self.temporal_attn(st_out)
        return st_out, coe1, coe2
    '''


    def get_total_loss(self,graphs, degrees, cen_var,split):
        mu1 = 0#0.2 #0-1  0.1
        mu2 = 0#0.2 #0-1  0.2
        loss, coe1 = self.get_loss(graphs,split)
        loss += mu1*self.get_spatial_loss(graphs, coe1, degrees,split)
        loss += mu2*self.get_temporal_loss(graphs, coe1, cen_var,split)
        return loss

    def get_loss(self, graphs, split):  #需要修改
        # run gnn
        final_emb, coe1 = self.forward(graphs)  # [N, T, F]
        self.graph_loss = 0
        for t in range(len(graphs)):
            y = graphs[t].y
            #y = y.numpy()
            y = torch.argmax(y, dim=1)
            #y = np.argmax(y, axis=1)
            #labels = torch.from_numpy(y).long().cuda()
            emb_t = final_emb[t]  # [N, F]
            y_hat = F.softmax(emb_t, dim=1)
            self.graph_loss += self.criterion(y_hat[:split].cuda(), y[:split])
        return self.graph_loss/len(graphs), coe1

    def get_spatial_loss(self, graphs, coe1, degrees, split):
        self.graph_spatial_loss = 0
        for t in range(len(graphs)):
            c1 = coe1[t]
            node_num = graphs[t].x.shape[0]
            #graph = to_networkx(graphs[t])
            #x = graphs[t].x
            edge_index = graphs[t].edge_index
            # x_origin = cp.deepcopy(x)
            #edge_weight = graphs[t].edge_attr.reshape(-1, 1)
            #node_num = x.shape[0]
            alph1 = scatter(c1, edge_index[0], dim=0, reduce="sum")
            alph1 = torch.norm(alph1, p=2, dim=1)
            sim = torch.cosine_similarity(graphs[t].x[edge_index[0]],graphs[t].x[edge_index[1]])#+1
            #sim = F.pairwise_distance(graphs[t].x[edge_index[0]][:-node_num],graphs[t].x[edge_index[1][:-node_num]], p=2)
            sim1 = scatter(sim, edge_index[1], dim=0, reduce="mean")
            #self.graph_spatial_loss += torch.mean((alph1 + alph2) / 2 - degrees[t])
            self.graph_spatial_loss += torch.mean((alph1-sim1/(degrees[t]+1))[:split])
        return self.graph_spatial_loss/len(graphs)

    def get_temporal_loss(self, graphs, coe1, cen_var, split):
        self.graph_temporal_loss = 0
        for t in range(len(graphs)):
            c1 = coe1[t]
            edge_index = graphs[t].edge_index
            # x_origin = cp.deepcopy(x)
            # edge_weight = graphs[t].edge_attr.reshape(-1, 1)
            # node_num = x.shape[0]
            alph1 = scatter(c1, edge_index[0], dim=0, reduce="sum")
            alph1 = torch.norm(alph1, p=2, dim=1)
            self.graph_temporal_loss += torch.mean(cen_var[t][:split] * torch.mean(alph1[:split]))
        return self.graph_temporal_loss/len(graphs)



class STG(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(STG, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, 32)
        self.conv_layer2 = GCNConv(32, 16)
        self.recurrent = LSTM(16,16)
        self.linear = torch.nn.Linear(16, self.num_classes)
        '''
        self.features = nn.Sequential(
            nn.Linear(self.node_features, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        '''
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H=None):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        h = F.relu(h)
        h, H = self.recurrent(h[None, :, :], H)
        h = F.relu(h)
        h = self.linear(h.squeeze())
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        return F.softmax(h, dim=1), H
        #out_x1 = self.features(x_1)
        #out_l = self.labels(label)
        #is_member = self.combine(torch.cat((out_x1, out_l), 1))
        #return self.output(is_member)


class GCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(GCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, 32)
        self.conv_layer2 = GCNConv(32, 16)
        self.linear = torch.nn.Linear(16, num_classes)
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H = None):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        return F.softmax(h, dim=1), H




class DCRNN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(DCRNN_RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)

    def forward(self, x, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1),H

class DCRNN_Attack(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(DCRNN_Attack, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        self.node_features = node_features
        self.num_classes = num_classes
        #self.recurrent2 = DCRNN(32, num_classes, 1)
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1), H

class DCRNN_STSA(torch.nn.Module):
    def __init__(self, args, node_features, num_classes, time_length):
        super(DCRNN_STSA, self).__init__()
        self.args = args
        self.num_time_steps = time_length
        self.num_features = node_features
        self.num_classes = num_classes
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.spatial = StructuralAttentionLayer(input_dim=node_features,
                                                output_dim=self.structural_layer_config[0],
                                                n_heads=self.structural_head_config[0],
                                                attn_drop=self.spatial_drop,
                                                ffd_drop=self.spatial_drop,
                                                residual=self.args.residual)
        self.recurrent = DCRNN(node_features, 32, 1)
        self.temporal = TemporalAttentionLayer2(input_dim=32,
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
        self.linear = torch.nn.Linear(32, num_classes)
        self.node_features = node_features
        self.num_classes = num_classes
        #self.recurrent2 = DCRNN(32, num_classes, 1)
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.spatial()
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1), H


class GConvGRU_Attack(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(GConvGRU_Attack, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        self.node_features = node_features
        self.num_classes = num_classes
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()
        #self.recurrent2 = DCRNN(32, num_classes, 1)

    def forward(self, x,y, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1),H

class GConvGRU_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(GConvGRU_RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1),H

class Dynamic_GNN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(Dynamic_GNN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        #self.recurrent = EvolveGCNO(node_features)
        #self.recurrent = TGCN(node_features, 32)
        #self.recurrent = GConvGRU(node_features, 32, 1)
        #self.recurrent = A3TGCN(node_features, 32, 1)
        #self.recurrent = GCLSTM(node_features, second_features, 1)
        #self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
    def forward(self, x, edge_index, edge_weight, H = None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1),H

class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, num_classes)
        #self.recurrent2 = EvolveGCNO(num_classes)

    def forward(self, x, edge_index, edge_weight, H = None):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
       #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1), H

class TGCN_Attack(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(TGCN_Attack, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        #self.recurrent2 = TGCN(32, num_classes)
        self.linear = torch.nn.Linear(32, num_classes)
        self.node_features = node_features
        self.num_classes = num_classes
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H=None):
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1), H

class TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(TGCN_RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        #self.recurrent2 = TGCN(32, num_classes)
        self.linear = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_weight, H=None):
        h = self.recurrent(x, edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1), H


class A3TGCN_Attack(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(A3TGCN_Attack, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        self.node_features = node_features
        self.num_classes = num_classes
        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.num_classes * 2, 2),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, y, edge_index, edge_weight, H=None):
        h = self.recurrent(x.view(x.shape[0], x.shape[1], 1), edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        y = self.labels(y)
        h = self.combine(torch.cat((h, y), 1))
        return F.softmax(h, dim=1), H


class A3TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(A3TGCN_RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)


    def forward(self, x, edge_index, edge_weight, H=None):
        h = self.recurrent(x.view(x.shape[0],x.shape[1],1), edge_index, edge_weight, H)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1), H







class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)




class VRecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(VRecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)


class ModelfreeGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(ModelfreeGCN, self).__init__()
        self.conv_layer = GCNConv(in_channels=node_features,
                                  out_channels=node_features,
                                  improved=False,
                                  cached=False,
                                  normalize=True,
                                  add_self_loops=True,
                                  bias=False)
        '''
        self.conv_layer2 = GCNConv(in_channels=node_features,
                                  out_channels=node_features,
                                  improved=False,
                                  cached=False,
                                  normalize=True,
                                  add_self_loops=True,
                                  bias=False)
        '''
    def forward(self, x, edge_index, edge_weight):
        Weight = self.conv_layer.lin.weight
        self.conv_layer.lin.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]))
        #Weight2 = self.conv_layer2.lin.weight
        #self.conv_layer2.lin.weight = Parameter(torch.eye(Weight2.shape[0], Weight2.shape[1]))
        h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        #h = self.conv_layer2(h, edge_index, edge_weight)

        #Weight = self.conv_layer.weight
        #self.conv_layer.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]))
        #h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return h





def data_preprossing(dataset):
    graphs = []
    for time, snapshot in enumerate(dataset):
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        edge_index_0 = torch.cat((edge_index[0], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        edge_index_1 = torch.cat((edge_index[1], torch.arange(snapshot.x.shape[0])), dim=0).tolist()
        new_edge_index = torch.tensor([edge_index_0,edge_index_1])
        new_edge_attr = torch.cat((edge_attr, torch.ones(snapshot.x.shape[0])), dim=0)
        graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=snapshot.y)
        graphs.append(graph)
    return graphs