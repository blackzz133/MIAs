import torch
import numpy as np
import networkx as nx
from dataloader import DBLPLoader
from torch_geometric_temporal.signal import temporal_signal_split,train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import random
from models import ModelfreeGCN, VRecurrentGCN, GCN, DCRNN_RecurrentGCN, \
    EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN, GConvGRU_RecurrentGCN, Robust_RGNN
from tqdm import tqdm
import torch.nn.functional as F
from deeprobust.graph.global_attack import NodeEmbeddingAttack, PGDAttack, DICE
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import  from_scipy_sparse_matrix, to_dense_adj
from pyvacy import optim, analysis
import torch_geometric
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import copy as cp
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader
#from sampler import SubsetSequentialSampler
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data.data import Data

def raw_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    url = str(dataname)+'/victim/'+str(victim_type)
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node*train_test_ratio)
    #train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)

    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.to(device)
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        #print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    #loader = DBLPLoader('DBLP5')
    #dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.train()
    for epoch in tqdm(range(500)):
        Hidden = None
        cost = 0
        for time, snapshot in enumerate(victim_loader):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #print(cost)
        cost = cost / (time + 1)
        o, _ = objective_function(model, victim_loader, split, device, type=None)
        # if cost < min_cost:
        # min_cost = cost
        # save_model = model
        #print('The '+str(epoch)+' training loss is '+str(cost))
        #print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()

    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(victim_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat[split:], labels[split:])
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': model.state_dict()}, f=url)
    objective_function(model, victim_loader, split, device, type='train')
    objective_function(model, victim_loader, split, device, type='test')
    return model

def relax_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    url = str(dataname)+'/victim/'+str(victim_type)+'-relaxloss'
    #train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.to(device)
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        #print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    #loader = DBLPLoader('DBLP5')
    #dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    use_cuda = torch.cuda.is_available()
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    model.train()
    relax_alpha = 0
    for epoch in tqdm(range(500)):
        Hidden = None
        cost = 0
        relax_loss = 0
        for time, snapshot in enumerate(victim_loader):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            with torch.no_grad():
                prob_gt = y_hat[torch.arange(y.shape[0]), torch.tensor(y)]
                prob_ngt = (1.0 - prob_gt) / (num_classes - 1)
                onehot = F.one_hot(torch.tensor(y), num_classes=num_classes).to(device)
                soft_labels = onehot * prob_gt.unsqueeze(-1).repeat(1, num_classes) \
                              + (1 - onehot) * prob_ngt.unsqueeze(-1).repeat(1, num_classes)
                soft_labels = torch.argmax(soft_labels, axis = 1)
            relax_loss += criterion(y_hat[:split], soft_labels[:split].to(device))
            # print(cost)
        cost = cost / (time + 1)
        print('The ' + str(epoch) + ' training loss is ' + str(cost))
        # print(cost)
        if cost >= relax_alpha:
            cost.backward()
        else:
            if epoch % 2 == 0:
                relax_loss = -cost
            relax_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()


    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': model.state_dict()}, f=url)
    objective_function(model, victim_loader, split, device, type='train')
    objective_function(model, victim_loader, split, device, type='test')
    return model


def adver_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    url = str(dataname)+'/victim/'+str(victim_type)+'-adver'
    perturbation_rate = 1.1
    #train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.to(device)
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        #print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    #loader = DBLPLoader('DBLP5')
    #dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    use_cuda = torch.cuda.is_available()
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    #train_loader, test_loader = temporal_signal_split(train_loader, train_test_ratio)
    adver_dataset = edge_attack_perturbation(victim_loader, perturbation_rate) #adversarial training
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.train()
    for epoch in tqdm(range(500)):
        Hidden = None
        cost = 0
        for time, snapshot in enumerate(adver_dataset):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #print(cost)
        cost = cost / (time + 1)
        print('The '+str(epoch)+' training loss is '+str(cost))
        #print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': model.state_dict()}, f=url)
    objective_function(model, victim_loader, split, device, type='train')
    objective_function(model, victim_loader, split, device, type='test')
    return model

def DP_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    url = str(dataname)+'/victim/'+str(victim_type)+'-DP2'
    #train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    time_len = torch.tensor(victim_loader.features).shape[0]
    noise_multiplier = 0.1
    max_grad_norm = 5.0
    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)

    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)

    if torch.cuda.is_available():
        model = model.to(device)
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        #print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    #loader = DBLPLoader('DBLP5')
    #dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    use_cuda = torch.cuda.is_available()
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #0.006
    #optim.DPAdam
    #optimizer = optim.DPSGD(params=model.parameters(), lr=lr, l2_norm_clip=1, noise_multiplier=1.1, minibatch_size=1, microbatch_size=time_len)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.train()
    for epoch in tqdm(range(500)):
        Hidden = None
        cost = 0
        optimizer.zero_grad()
        for time, snapshot in enumerate(victim_loader):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
            cost = criterion(y_hat[:split], labels[:split])
            #print(cost)
        print('The '+str(epoch)+' training loss is '+str(cost))
        cost.backward()
        for param in model.parameters():
            if param.grad is not None:
                # sensitivity calculation
                grad_norm = torch.norm(param.grad)
                layer_max_grad_norm = max_grad_norm / len(list(model.parameters()))
                clip_coef = layer_max_grad_norm / (grad_norm + 1e-6)
                if clip_coef < 1:
                    param.grad.data.mul_(clip_coef)
                # add noise
                noise = torch.normal(0, noise_multiplier * layer_max_grad_norm, size=param.grad.shape)
                param.grad.data.add_(noise.cuda())
        optimizer.step()
        optimizer.zero_grad()
        #print(cost)
        #optimizer.step()
        #optimizer.zero_grad()
        #scheduler.step()
    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': model.state_dict()}, f=url)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    objective_function(model, victim_loader, split, device, type='train')
    objective_function(model, victim_loader, split, device, type='test')
    return model

def STSA_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    #url = str(dataname)+'/victim/'+str(victim_type)+'-STSA'
    url = str(dataname) + '/victim/' + str(victim_type) + '-STSA'
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    # train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    time_length = torch.tensor(victim_loader.features).shape[0]
    model = Robust_RGNN(args, device, node_features, time_length, num_classes, victim_type)
    victim_graphs = data_preprossing(victim_loader)
    if torch.cuda.is_available():
        model = model.to(device)
    # exit()
    # model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        # print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    # loader = DBLPLoader('DBLP5')
    # dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    # train_loader = dataset
    # print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    # print(train_loader.features)
    # print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    time = len(victim_graphs)
    degrees = []
    # centrality
    cens = []
    cen_var = []
    for t in range(time):
        graph = to_networkx(victim_graphs[t])
        degree = torch.tensor(list(dict(graph.degree).values()))
        degrees.append(degree.cuda())
        cen = torch.tensor(list(dict(nx.closeness_centrality(graph)).values()))
        cens.append(cen.cuda())
    cen_var.append(torch.zeros(cens[0].shape).cuda())
    for t in range(1, time - 1):
        cen_var.append((abs(cens[t] - cens[t - 1]) + abs(cens[t + 1] - cens[t])).cuda())
    cen_var.append(torch.zeros(cens[time - 1].shape).cuda())
    model.train()
    cost = 0
    max_o = 0
    min_cost = 100
    save_model = cp.deepcopy(model)
    for epoch in tqdm(range(300)):
        cost = model.get_total_loss(victim_graphs, degrees, cen_var, split)
        o,_ = objective_function2(model, victim_loader, split, device, type=None)
        #if cost < min_cost:
            #min_cost = cost
            #save_model = model
        if o > max_o:
           max_o = o
           save_model = model
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))

            # print(cost)
        # print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(victim_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat[split:], labels[split:])
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': save_model.state_dict()}, f=url)
    objective_function2(model, victim_loader, split, device, type='train')
    objective_function2(model, victim_loader, split, device, type='test')
    return model

def DPSTSA_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    #url = str(dataname)+'/victim/'+str(victim_type)+'-STSA'
    url = str(dataname) + '/victim/' + str(victim_type) + '-DPSTSA'
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    # train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    time_length = torch.tensor(victim_loader.features).shape[0]
    model = Robust_RGNN(args, device, node_features, time_length, num_classes, victim_type)
    victim_graphs = data_preprossing(victim_loader)
    if torch.cuda.is_available():
        model = model.to(device)
    # exit()
    # model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        # print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    # loader = DBLPLoader('DBLP5')
    # dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    time = len(victim_graphs)
    degrees = []
    # centrality
    cens = []
    cen_var = []
    for t in range(time):
        graph = to_networkx(victim_graphs[t])
        degree = torch.tensor(list(dict(graph.degree).values()))
        degrees.append(degree.cuda())
        cen = torch.tensor(list(dict(nx.closeness_centrality(graph)).values()))
        cens.append(cen.cuda())
    cen_var.append(torch.zeros(cens[0].shape).cuda())
    for t in range(1, time - 1):
        cen_var.append((abs(cens[t] - cens[t - 1]) + abs(cens[t + 1] - cens[t])).cuda())
    cen_var.append(torch.zeros(cens[time - 1].shape).cuda())
    model.train()
    cost = 0
    max_o = 0
    min_cost = 100
    save_model = cp.deepcopy(model)
    spatial_sensitivity, temporal_sensitivity = compute_layer_gradient_sensitivity(model, victim_graphs)
    #the privacy budget epsilon and exponential decay rate lamba
    epsilon = 1 #0.1 to 1
    clipping_threshold =5
    max_grad_norm = 1
    lamba = 1
    spatial_budget, temporal_budget = budget_weights_calculation(torch.tensor([spatial_sensitivity, temporal_sensitivity]), epsilon, lamba)
    for epoch in tqdm(range(300)):
        cost = model.get_total_loss(victim_graphs, degrees, cen_var, split)
        o,_ = objective_function2(model, victim_loader, split, device, type=None)
        #if cost < min_cost:
            #min_cost = cost
            #save_model = model
        if o > max_o:
           max_o = o
           save_model = model
        if (epoch + 1) % 20 == 0:
            print('The ' + str(epoch) + ' training loss is ' + str(cost))
        cost.backward()

        # Gradient clipping
        if ((epoch + 1) % 20 == 0) and epoch<=280:
            # add gaussian noise
            for name, params in model.named_parameters():
                if params.grad is not None:
                    grad_norm = torch.norm(params.grad)
                    layer_max_grad_norm = max_grad_norm / len(list(model.parameters()))
                    clip_coef = layer_max_grad_norm / (grad_norm + 1e-6)
                    if clip_coef < 1:
                        params.grad.data.mul_(clip_coef)
                    #torch.nn.utils.clip_grad_norm_(model.structural_attn1.parameters(), clipping_threshold)
                    #torch.nn.utils.clip_grad_norm_(model.temporal_attn.parameters(), clipping_threshold)
                    if 'structural_attn1' in name:
                        #params.grad = params.grad
                        params.grad = add_noise(params.grad, spatial_sensitivity, spatial_budget)
                    elif 'temporal_attn' in name:
                        params.grad = add_noise(params.grad, temporal_sensitivity, temporal_budget)
                        #params.grad = params.grad


        '''
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                layer_max_grad_norm = max_grad_norm / len(list(model.parameters()))
                clip_coef = layer_max_grad_norm / (grad_norm + 1e-6)
                if clip_coef < 1:
                    param.grad.data.mul_(clip_coef)
                # 添加噪声
                noise_multiplier_spatial = spatial_sensitivity / spatial_budget
                noise_multiplier_temporal = temporal_sensitivity / temporal_budget
                if 'structural_attn1' in name:
                    noise_multiplier = noise_multiplier_spatial
                elif 'temporal_attn' in name:
                    noise_multiplier = noise_multiplier_temporal
                else:
                    continue  # 如果不是这两层的参数则跳过
                noise = torch.normal(0, noise_multiplier * layer_max_grad_norm, size=param.grad.shape)
                param.grad.data.add_(noise.cuda())
        '''
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
    model.eval()
    '''
    cost = 0
    Hidden = None
    for time, snapshot in enumerate(victim_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat[split:], labels[split:])
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    '''
    torch.save({'victim_model': save_model.state_dict()}, f=url)
    objective_function2(model, victim_loader, split, device, type='train')
    objective_function2(model, victim_loader, split, device, type='test')
    return model





def objective_function(model, dataset, split, device, type):
    accuracy = 0
    f1 = 0
    total_time = 0
    Hidden = None
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            output, Hidden = model.to(device)(x, edge_index, edge_attr, Hidden)
            victim_labels = torch.argmax(output.detach(), dim=1).long().clone().to(device)
            if type == 'train':
                accuracy += torch.eq(y_labels[:split], victim_labels[:split]).sum() / y_labels[:split].shape[0]
                f1 += f1_score(y_labels[:split].to('cpu'), victim_labels[:split].to('cpu'),average='weighted')

            elif type =='test':
                accuracy += torch.eq(y_labels[split:], victim_labels[split:]).sum() / y_labels[split:].shape[0]
                f1 += f1_score(y_labels[split:].to('cpu'), victim_labels[split:].to('cpu'),average='weighted')
            elif type == None:
                accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
                f1 += f1_score(y_labels.to('cpu'), victim_labels.to('cpu'),average='weighted')

            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))
    return accuracy / total_time, f1/total_time

def objective_function2(model, dataset, split, device, type):
    model.eval
    y, coe1 = model(dataset)
    accuracy = 0
    total_time = 0
    f1=0
    Hidden = None
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            y_hat = torch.argmax(y[time], axis=1).cuda()
            y_label = torch.argmax(snapshot.y, axis=1).cuda()
            #accuracy += torch.eq(y_hat, y_label).sum() / y_hat.shape[0]
            if type == 'train':
                accuracy += torch.eq(y_label[:split], y_hat[:split]).sum() / y_label[:split].shape[0]
                f1 += f1_score(y_label[:split].to('cpu'), y_hat[:split].to('cpu'),average='weighted')
            elif type =='test':
                accuracy += torch.eq(y_label[split:], y_hat[split:]).sum() / y_label[split:].shape[0]
                f1 += f1_score(y_label[split:].to('cpu'),  y_hat[split:].to('cpu'),average='weighted')
            elif type == None:
                accuracy += torch.eq(y_label, y_hat).sum() / y_label.shape[0]
                f1 += f1_score(y_label.to('cpu'),  y_hat.to('cpu'), average='weighted')
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))
    return accuracy / total_time, f1 / total_time

def edge_attack_perturbation(train_loader, perturbation_rate):
    edge_indices = []
    edge_weights = []

    for time, snapshot in enumerate(train_loader):
        adj = to_scipy_sparse_matrix(snapshot.edge_index, snapshot.edge_attr).tocsr()
        features = snapshot.x
        labels = torch.argmax(snapshot.y, dim=1)
        n_perturbations = round(snapshot.x.shape[0] * perturbation_rate)
        model =DICE()
        model.attack(adj, labels, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
        # w = (adj!=new_a).nnz==0
        # rand.attack(adj, attack_type="add", n_candidates=10000)
        # rand.attack(adj, attack_type="add_by_remove", n_candidates=10000)
        new_edge_index, new_edge_attr = from_scipy_sparse_matrix(modified_adj)
        # edge_indices.append(new_edge_index.tolist())
        # edge_weights.append(new_edge_attr.tolist())
        edge_indices.append(new_edge_index.tolist())
        edge_weights.append(new_edge_attr.tolist())
    train_loader.edge_indices = edge_indices
    train_loader.edge_weights = edge_weights
    return train_loader

def data_preprossing2(dataset):
    graphs = []
    for time, snapshot in enumerate(dataset):
        graph = torch_geometric.data.data.Data(x=snapshot.x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr, y=snapshot.y)
        graphs.append(graph)
    return graphs

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


def add_noise(gradient, sensitivity, epsilon):
    # Calculate standard deviation of Gaussian noise
    sigma = sensitivity / epsilon
    # Generate Gaussian noise
    noise = torch.distributions.Laplace(loc=0.0, scale=sigma).sample(gradient.shape)
    #noise =torch.normal(mean=0.0, std=sigma, size=gradient.shape)
    noisy_gradient = gradient.cuda() + noise.cuda()
    return noisy_gradient


    # Add noise to gradient





def compute_layer_gradient_sensitivity(model, graph_list, num_samples=10):
    sensitivities = {'spatial': [], 'temporal': []}
    for i, graph in enumerate(graph_list):
        num_nodes = graph.x.size(0)
        max_spatial_sensitivity = 0
        max_temporal_sensitivity = 0

        sampled_nodes = random.sample(range(num_nodes), min(num_samples, num_nodes))

        for node_to_remove in sampled_nodes:
            # Perform forward pass on the full graph
            output_full,_ = model([graph])

            # Compute gradients for spatial and temporal layers for the full graph
            node_output_full = output_full[:, node_to_remove, :].mean()
            grad_spatial_full = torch.autograd.grad(node_output_full, model.structural_attn1.parameters(), allow_unused=True, create_graph=False, retain_graph=True)
            grad_temporal_full = torch.autograd.grad(node_output_full, model.temporal_attn.parameters(), allow_unused=True, create_graph=False, retain_graph=True)

            # Remove the node from the graph by creating a mask
            mask = torch.ones(num_nodes, dtype=torch.bool)
            mask[node_to_remove] = False
            reduced_graph_x = graph.x[mask]
            reduced_graph_edge_index, reduced_graph_edge_attr = pyg_utils.subgraph(mask, graph.edge_index, graph.edge_attr, relabel_nodes=True)

            # Forward pass with node removed
            reduced_graph = Data(x=reduced_graph_x, edge_index=reduced_graph_edge_index, edge_attr=reduced_graph_edge_attr)
            reduced_graph.x.requires_grad = True
            output_reduced,_ = model([reduced_graph])

            # Compute gradients for spatial and temporal layers after node removal
            node_output_reduced = output_reduced.mean()
            grad_spatial_reduced = torch.autograd.grad(node_output_reduced, model.structural_attn1.parameters(), allow_unused=True, create_graph=False, retain_graph=True)
            grad_temporal_reduced = torch.autograd.grad(node_output_reduced, model.temporal_attn.parameters(), allow_unused=True, create_graph=False, retain_graph=True)

            # Calculate sensitivity for spatial and temporal layers
            spatial_sensitivity = torch.norm(torch.cat([torch.norm(gf - gr).unsqueeze(0) for gf, gr in zip(grad_spatial_full, grad_spatial_reduced)]), p=float('inf'))
            temporal_sensitivity = torch.norm(torch.cat([torch.norm(gf - gr).unsqueeze(0) for gf, gr in zip(grad_temporal_full[1], grad_temporal_reduced[1])]), p=float('inf'))

            # Track the maximum sensitivity across sampled nodes
            max_spatial_sensitivity = max(max_spatial_sensitivity, spatial_sensitivity)
            max_temporal_sensitivity = max(max_temporal_sensitivity, temporal_sensitivity)

        sensitivities['spatial'].append(max_spatial_sensitivity)
        sensitivities['temporal'].append(max_temporal_sensitivity)

    # Aggregate sensitivities across all data for each layer using the min value
    final_spatial_sensitivity = torch.mean(torch.tensor(sensitivities['spatial']))
    final_temporal_sensitivity = torch.mean(torch.tensor(sensitivities['temporal']))

    return final_spatial_sensitivity, final_temporal_sensitivity



def budget_weights_calculation(sensitivities, epsilon, lamba):
    weights = torch.exp(- lamba * sensitivities)
    normalized_weights = weights / weights.sum()
    privacy_budgets = normalized_weights * epsilon
    return privacy_budgets[0], privacy_budgets[1]

def clip_gradients(parameters, max_norm):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)




