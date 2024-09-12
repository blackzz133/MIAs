import torch
import numpy as np
import networkx as nx
from dataloader import DBLPELoader,DBLPLoader
from models import  GCN, STG, DCRNN_Attack, GConvGRU_Attack, TGCN_Attack,A3TGCN_Attack
from torch_geometric_temporal.signal import temporal_signal_split, train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from ModelExtraction.active_learning import *
import copy as cp
import torch.nn.functional as F
#from config import *
import copy as cp
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.cluster import KMeans
import networkx as nx
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import math


class raw_attack_model:
    def __init__(self, model_type, defence_type, args, dataname, victim_loader, shadow_loader, victim_model, shadow_models, node_features, num_classes, attack_type, device):
        self.model_type = model_type
        self.defence_type = defence_type
        self.args = args
        self.dataname = dataname
        self.victim_loader = victim_loader
        self.shadow_loader = shadow_loader
        self.attack_type = attack_type
        self.device = device
        self.victim_model = victim_model.to(self.device)
        self.shadow_models = shadow_models
        if model_type == 'STG':
            self.attack_model = STG(node_features=node_features,
                                    num_classes=num_classes).to(self.device)
        if model_type == 'GCN':
            self.attack_model = GCN(node_features=node_features,
                                    num_classes=num_classes).to(self.device)
        if model_type == 'DCRNN':
            self.attack_model = DCRNN_Attack(node_features=node_features,
                                    num_classes=num_classes).to(self.device)
        if model_type == 'GConvGRU':
            self.attack_model = GConvGRU_Attack(node_features=node_features,
                                    num_classes=num_classes).to(self.device)
        if model_type == 'TGCN':
            self.attack_model = TGCN_Attack(node_features=node_features,
                                    num_classes=num_classes).to(self.device)
        if model_type == 'A3TGCN':
            self.attack_model = A3TGCN_Attack(node_features=node_features,
                                    num_classes=num_classes).to(self.device)

        self.shadow_model1 = self.shadow_models['model1'].to(self.device)
        self.shadow_model2 = self.shadow_models['model2'].to(self.device)
        self.shadow_model3 = self.shadow_models['model3'].to(self.device)
        self.shadow_model4 = self.shadow_models['model4'].to(self.device)


    def fit(self, loader, train_test_ratio, num_epoches, learning_rate, criterion):
        '''
        if self.defence_type == 'raw':
            url = str(self.dataname) + '/attack/'+str(self.model_type)+'/'+str(self.defence_type)+'/' + str(self.attack_type)
        else:
            url = str(self.dataname) + '/attack/'+str(self.model_type)+ '/' + str(self.defence_type) + '/' + str(self.attack_type)
        '''
        #url = str(self.dataname) + '/attack/' + str(self.model_type) + '/' + str(self.defence_type) + '/' + str(
            #self.attack_type)
        url = str(self.dataname) + '/attack/' + str(self.model_type) + '/' + str(self.defence_type) + '/' + str(
            self.attack_type)
        ratio = 0.5  # for testing
        if os.path.exists(url) == False:
            file = open(url, 'w')
        if os.path.getsize(url) > 0:
            print('Saved attack model is loaded')
            weights = torch.load(f=url)
            self.attack_model.load_state_dict(weights['attack_model'], strict=False)
            # print(weights['victim_model'])
            return
        lr = learning_rate
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=lr)  # 0.006
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        H1 = None
        H2 = None
        H12 = None
        H22 = None
        H13 = None
        H23 = None
        H14 = None
        H24 = None
        All_X= None
        # get positive output from shadow model
        for time, snapshot in enumerate(loader):
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device)
            split = round(snapshot.y.shape[0]*train_test_ratio)
            split2 = snapshot.y.shape[0]-split
            output, H1 = self.shadow_model1(snapshot.x.to(self.device), edge_index, edge_attr, H1)
            output2, H12 = self.shadow_model2(snapshot.x.to(self.device), edge_index, edge_attr, H12)
            output3, H13 = self.shadow_model3(snapshot.x.to(self.device), edge_index, edge_attr, H13)
            output4, H14 = self.shadow_model4(snapshot.x.to(self.device), edge_index, edge_attr, H14)
            X1 = output[:split].clone().detach().to(self.device)
            X2 = output[split:].clone().detach().to(self.device)
            X1[split // 4:split // 2] = output2[split // 4:split // 2].clone().detach().to(self.device)
            X2[split2 // 4:split2 // 2] = output2[split+split2 // 4:split+split2 // 2].clone().detach().to(self.device)
            X1[split // 2:split * 3 // 4] = output3[split // 2:split * 3 // 4].clone().detach().to(
                self.device)
            X2[split2 // 2:split2 * 3 // 4] = output3[split+split2 // 2:split+split2 * 3 // 4].clone().detach().to(
                self.device)
            X1[split * 3 // 4:split] = output4[split * 3 // 4:split].clone().detach().to(self.device)
            X2[split2 * 3 // 4:] = output4[split+split2 * 3 // 4:].clone().detach().to(self.device)
            X = torch.cat((X1, X2), dim=0)[None, :, :]

            try:
                All_X = torch.cat((All_X, X))
            except:
                All_X = X

        # train
        max_f1 = 0
        epoch_id = 0
        for epoch in tqdm(range(num_epoches)):
            Hidden1 = None
            Hidden2 = None
            pos_loss = 0
            neg_loss = 0
            # postive samples loss
            for time, snapshot in enumerate(loader):
                # labels = torch.ones(snapshot.y.shape[0]).to(self.device)  #这里
                y = snapshot.y.to(self.device)
                split = round(snapshot.y.shape[0] * train_test_ratio)
                labels = torch.zeros(snapshot.y.shape[0], 2).to(self.device)
                labels[:split, 1] = 1
                labels[split:, 0] = 1
                edge_index = snapshot.edge_index.to(self.device)
                edge_attr = snapshot.edge_attr.to(self.device)
                h, Hidden1 = self.attack_model(All_X[time], y, edge_index, edge_attr, Hidden1)
                pos_loss += criterion(h[:split], labels[:split])
                neg_loss += criterion(h[split:], labels[split:])
            pos_loss = pos_loss / (time + 1)
            neg_loss = neg_loss / (time + 1)
            cost = pos_loss + neg_loss
            if epoch % 20 == 0:
                print('The ' + str(epoch) + ' training loss is ' + str(cost))
                if epoch > 0:
                    if (self.defence_type=='STSA') or (self.defence_type=='DPSTSA'):
                        result, accuracy = self.infer2(self.shadow_loader, train_test_ratio)
                    else:
                        result, accuracy = self.infer(self.shadow_loader, train_test_ratio)
                    total = result+accuracy
                    if (total > max_f1) and (epoch > 0):
                        attack_model = cp.deepcopy(self.attack_model)
                        max_f1 = total
                        epoch_id = epoch
            # print(cost)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()
        self.attack_model = cp.deepcopy(attack_model)
        torch.save({'attack_model': attack_model.state_dict()}, f=url)
        print('epoch_id is: '+str(epoch_id))

    def infer(self, dataloader, ratio):
        Hidden = None
        Hidden2 =None
        accuracy = 0
        total_time = 0
        precision = 0
        all_labels = None
        all_attack_labels = None
        node_num = torch.tensor(dataloader.features).shape[1]
        split = round(node_num*ratio)
        for time, snapshot in enumerate(dataloader):
            y = snapshot.y.to(self.device)
            labels = torch.zeros(snapshot.y.shape[0],2).to(self.device)
            labels[:split, 1] = 1
            labels[split:, 0] = 1
            labels = torch.argmax(labels, dim=1).to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device)
            x, Hidden = self.victim_model(snapshot.x.to(self.device), edge_index, edge_attr, Hidden)
            h, Hidden2 = self.attack_model(x, y, edge_index, edge_attr, Hidden2)
            attack_labels = torch.argmax(h, dim=1).long().clone().to(self.device)
            try:
                all_labels = torch.cat((all_labels, labels))
                all_attack_labels = torch.cat((all_attack_labels, attack_labels))
            except:
                all_labels = labels
                all_attack_labels = attack_labels
        #result1 = recall_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        #print('The recall of membership inference is ' + str(result1))
        #result2 = precision_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        #print('The precision of membership inference is ' + str(result2))
        result = f1_score(all_labels.to('cpu'), all_attack_labels.to('cpu'), average='weighted')
        print('The f1-score of membership inference is ' + str(result))
        accuracy = accuracy_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        print('The accuracy of membership inference is ' + str(accuracy))

        return result, accuracy#, result1, result2
    def infer2(self, dataloader, ratio):
        Hidden = None
        Hidden2 =None
        accuracy = 0
        total_time = 0
        precision = 0
        all_labels = None
        all_attack_labels = None
        node_num = torch.tensor(dataloader.features).shape[1]
        X, coe1 = self.victim_model(dataloader)
        split = round(node_num*ratio)
        for time, snapshot in enumerate(dataloader):
            y = snapshot.y.to(self.device)
            labels = torch.zeros(snapshot.y.shape[0],2).to(self.device)
            labels[:split, 1] = 1
            labels[split:, 0] = 1
            labels = torch.argmax(labels, dim=1).to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device)
            #x, Hidden = self.victim_model(snapshot.x.to(self.device), edge_index, edge_attr, Hidden)
            h, Hidden2 = self.attack_model(X[time], y, edge_index, edge_attr, Hidden2)
            attack_labels = torch.argmax(h, dim=1).long().clone().to(self.device)
            try:
                all_labels = torch.cat((all_labels, labels))
                all_attack_labels = torch.cat((all_attack_labels, attack_labels))
            except:
                all_labels = labels
                all_attack_labels = attack_labels
        #result1 = recall_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        #print('The recall of membership inference is ' + str(result1))
        #result2 = precision_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        #print('The precision of membership inference is ' + str(result2))
        result = f1_score(all_labels.to('cpu'), all_attack_labels.to('cpu'), average='weighted')
        print('The f1-score of membership inference is ' + str(result))
        accuracy = accuracy_score(all_labels.to('cpu'), all_attack_labels.to('cpu'))
        print('The accuracy of membership inference is ' + str(accuracy))

        return result, accuracy#, result1, result2


