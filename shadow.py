import torch
import numpy as np
import networkx as nx
from dataloader import DBLPLoader
from torch_geometric_temporal.signal import temporal_signal_split,train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from models import ModelfreeGCN, VRecurrentGCN, GCN,LitDiffConvModel, DCRNN_RecurrentGCN, \
    EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN, GConvGRU_RecurrentGCN
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from torch.utils.data import DataLoader
#from sampler import SubsetSequentialSampler
def raw_shadow_model(args, dataname, shadow_type, shadow_loader, train_test_ratio, lr, device, victim_model):
    url = str(dataname)+'/shadow/'+str(shadow_type)
    node_features = torch.tensor(shadow_loader.features).shape[2]
    num_classes = torch.tensor(shadow_loader.targets).shape[2]
    num_node = torch.tensor(shadow_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    split1 = split//4
    split2 = split//2
    split3 = split*3//4
    '''

    DCRNN = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
    EvolveGCNO = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
    TGCN = TGCN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
    A3TGCN = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    '''
    if shadow_type == 'DCRNN':
        model1 = DCRNN_RecurrentGCN(node_features=node_features,
                                   num_classes=num_classes)
        model2 = DCRNN_RecurrentGCN(node_features=node_features,
                                        num_classes=num_classes)
        model3 = DCRNN_RecurrentGCN(node_features=node_features,
                                  num_classes=num_classes)
        model4 = DCRNN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    if shadow_type =='EVOLVEGCNO':
        model1 = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model2 = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model3 = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model4 = EvolveGCNO_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
    if shadow_type =='GConvGRU':
        model1 = GConvGRU_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model2 = GConvGRU_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model3 = GConvGRU_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model4 = GConvGRU_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)

    if shadow_type =='TGCN':
        model1 = TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model2 = TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model3 = TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model4 = TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)

    if shadow_type =='A3TGCN':
        model1 = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model2 = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model3 = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)
        model4 = A3TGCN_RecurrentGCN(node_features=node_features,
                                    num_classes=num_classes)


    models = {}
    if torch.cuda.is_available():
        model1 = model1.to(device)
        model2 = model2.to(device)
        model3 = model3.to(device)
        model4 = model4.to(device)
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved shadow model is loaded')
        weights = torch.load(f=url)
        model1.load_state_dict(weights['model1'], strict=False)
        model2.load_state_dict(weights['model2'], strict=False)
        model3.load_state_dict(weights['model3'], strict=False)
        model4.load_state_dict(weights['model4'], strict=False)
        #print(weights['victim_model'])
        models['model1'] = model1
        models['model2'] = model2
        models['model3'] = model3
        models['model4'] = model4
        return models

    #loader = DBLPLoader('DBLP5')
    #dataset = loader.get_dataset()
    print('The data of shadow_model is loaded')
    use_cuda = torch.cuda.is_available()
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr)
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    #scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=250, gamma=0.5)
    #scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=250, gamma=0.5)
    #scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=250, gamma=0.5)
    #scheduler4 = lr_scheduler.StepLR(optimizer4, step_size=250, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model1.train()
    for epoch in tqdm(range(300)):
        cost = 0
        Hidden0 = None
        Hidden = None
        for time, snapshot in enumerate(shadow_loader):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
            #y = y.detach().cpu()
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            y_hat, Hidden = model1(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #cost = cost + criterion(y_hat, labels)
            #print(cost)
        cost = cost / (time + 1)
        print('The '+str(epoch)+' training loss is '+str(cost))
        #print(cost)
        cost.backward()
        optimizer1.step()
        optimizer1.zero_grad()
    model1.eval()
    '''
    cost = 0
    Hidden0 = None
    Hidden = None
    for time, snapshot in enumerate(shadow_loader):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
        #y = y.detach().cpu()
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        y_hat,Hidden = model1(x, edge_index, edge_weight,Hidden)
        cost = cost + criterion(y_hat[:num_node//4], labels[:num_node//4])
        #cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("model1 Cross_Entropy: {:.4f}".format(cost))
    '''
    objective_function(model1, shadow_loader, split, device, type='train')
    objective_function(model1, shadow_loader, split, device, type='test')

    model2.train()
    for epoch in tqdm(range(300)):
        cost = 0
        Hidden0 = None
        Hidden = None
        for time, snapshot in enumerate(shadow_loader):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
            #y = y.detach().cpu()
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            y_hat, Hidden = model2(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        print('The ' + str(epoch) + ' training loss is ' + str(cost))
        # print(cost)
        cost.backward()
        optimizer2.step()
        optimizer2.zero_grad()
    model2.eval()
    '''
    cost = 0
    Hidden0 = None
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
        #y = y.detach().cpu()
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        y_hat, Hidden = model2(x, edge_index, edge_weight, Hidden)
        #cost = cost + criterion(y_hat[num_node//4:num_node//2], labels[num_node//4:num_node//2])
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("model2 Cross_Entropy: {:.4f}".format(cost))
    '''
    objective_function(model2, shadow_loader, split, device, type='train')
    objective_function(model2, shadow_loader, split, device, type='test')

    model3.train()
    for epoch in tqdm(range(300)):
        cost = 0
        Hidden0 = None
        Hidden = None
        for time, snapshot in enumerate(shadow_loader):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
            #y = y.detach().cpu()
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            y_hat, Hidden = model3(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        print('The ' + str(epoch) + ' training loss is ' + str(cost))
        # print(cost)
        cost.backward()
        optimizer3.step()
        optimizer3.zero_grad()
    model3.eval()
    '''
    cost = 0
    Hidden0 = None
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
        #y = y.detach().cpu()
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        y_hat, Hidden = model3(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat[num_node//2:num_node*3//4], labels[num_node//2:num_node*3//4])
        #cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("model3 Cross_Entropy: {:.4f}".format(cost))
    '''
    objective_function(model3, shadow_loader, split, device, type='train')
    objective_function(model3, shadow_loader, split, device, type='test')

    model4.train()
    for epoch in tqdm(range(300)):
        cost = 0
        Hidden0 = None
        Hidden = None
        for time, snapshot in enumerate(shadow_loader):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)
            #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
            #y = y.detach().cpu()
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().to(device)
            y_hat, Hidden = model4(x, edge_index, edge_weight, Hidden)
            cost = cost + criterion(y_hat[:split], labels[:split])
            #cost = cost + criterion(y_hat, labels)
            # print(cost)
        cost = cost / (time + 1)
        print('The ' + str(epoch) + ' training loss is ' + str(cost))
        # print(cost)
        cost.backward()
        optimizer4.step()
        optimizer4.zero_grad()
    model4.eval()
    '''
    cost = 0
    Hidden0 = None
    Hidden = None
    for time, snapshot in enumerate(test_loader):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device)
        #y, Hidden0 = victim_model(x,edge_index,edge_weight, Hidden0)
        #y = y.detach().cpu()
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().to(device)
        y_hat, Hidden = model4(x, edge_index, edge_weight, Hidden)
        cost = cost + criterion(y_hat[num_node*3//4:], labels[num_node*3//4:])
        #cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("model4 Cross_Entropy: {:.4f}".format(cost))
    '''
    objective_function(model4, shadow_loader, split, device, type='train')
    objective_function(model4, shadow_loader, split, device, type='test')
    torch.save({'model1': model1.state_dict(),'model2': model2.state_dict(),'model3': model3.state_dict(),'model4': model4.state_dict()}, f=url)
    models = {'model1':model1, 'model2':model2, 'model3':model3, 'model4': model4}
    return models


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
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))


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