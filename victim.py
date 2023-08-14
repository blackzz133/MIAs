import torch
import numpy as np
import networkx as nx
from dataloader import DBLPLoader
from torch_geometric_temporal.signal import temporal_signal_split,train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from models import ModelfreeGCN, VRecurrentGCN, GCN,LitDiffConvModel, DCRNN_RecurrentGCN, \
    EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN, GConvGRU_RecurrentGCN, Robust_RGNN
from tqdm import tqdm
import torch.nn.functional as F
from deeprobust.graph.global_attack import NodeEmbeddingAttack, PGDAttack, DICE
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import  from_scipy_sparse_matrix, to_dense_adj
from pyvacy import optim, analysis,sampling
import torch_geometric
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import copy as cp
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from torch.utils.data import DataLoader
#from sampler import SubsetSequentialSampler

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
    perturbation_rate = 1
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
    url = str(dataname)+'/victim/'+str(victim_type)+'-DP'
    #train_loader, test_loader = temporal_signal_split(victim_loader, train_test_ratio)
    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    time_len = torch.tensor(victim_loader.features).shape[0]
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
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.006
    #optim.DPAdam
    optimizer = optim.DPSGD(params=model.parameters(), lr=lr, l2_norm_clip=1, noise_multiplier=1.1, minibatch_size=1, microbatch_size=time_len)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.train()
    for epoch in tqdm(range(500)):
        Hidden = None
        cost = 0
        optimizer.zero_grad()
        for time, snapshot in enumerate(victim_loader):
            optimizer.zero_microbatch_grad()
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)
            cost = criterion(y_hat[:split], labels[:split])
            cost.backward()
            optimizer.microbatch_step()
            #print(cost)
        print('The '+str(epoch)+' training loss is '+str(cost))
        optimizer.step()
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
    url = str(dataname)+'/victim/'+str(victim_type)+'-STSA'
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
        o = objective_function2(model, victim_loader, split, device, type=None)
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
    #objective_function2(model, victim_loader, split, device, type='train')
    #objective_function2(model, victim_loader, split, device, type='test')
    return model





def objective_function(model, dataset, split, device, type):
    accuracy = 0
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
            elif type =='test':
                accuracy += torch.eq(y_labels[split:], victim_labels[split:]).sum() / y_labels[split:].shape[0]
            elif type == None:
                accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]

            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))
    return accuracy / total_time

def objective_function2(model, dataset, split, device, type):
    model.eval
    y, coe1 = model(dataset)
    accuracy = 0
    total_time = 0
    Hidden = None
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            y_hat = torch.argmax(y[time], axis=1).cuda()
            y_label = torch.argmax(snapshot.y, axis=1).cuda()
            #accuracy += torch.eq(y_hat, y_label).sum() / y_hat.shape[0]
            if type == 'train':
                accuracy += torch.eq(y_label[:split], y_hat[:split]).sum() / y_label[:split].shape[0]
            elif type =='test':
                accuracy += torch.eq(y_label[split:], y_hat[split:]).sum() / y_label[split:].shape[0]
            elif type == None:
                accuracy += torch.eq(y_label, y_hat).sum() / y_label.shape[0]
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))
    return accuracy / total_time

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


# 1.0704 lr=0.01 DCRNN
#The accuracy result of rgcn is tensor(0.8425, device='cuda:0')
#The accuracy result of rgcn is tensor(0.8355, device='cuda:0')

# 1.1102 lr=0.008 EVOLVE
#The accuracy result of rgcn is tensor(0.7986, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7940, device='cuda:0')

#1.1184  lr=0.01 TGCN
#The accuracy result of rgcn is tensor(0.7910, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7867, device='cuda:0')

#1.1215  lr=0.01 A3TGCN
# The accuracy result of rgcn is tensor(0.7910, device='cuda:0')
# The accuracy result of rgcn is tensor(0.7838, device='cuda:0')