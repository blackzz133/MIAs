import torch
import numpy as np
import networkx as nx
from dataloader import DBLPLoader
import argparse
from victim import raw_victim_model, relax_victim_model, adver_victim_model, DP_victim_model, objective_function
from attack import raw_attack_model
from shadow import raw_shadow_model
from sklearn.metrics import f1_score
from torch_geometric_temporal.signal import temporal_signal_split,train_test_split

#from TGCN.signal.train_test_split import temporal_signal_split
import os


parser = argparse.ArgumentParser()


#def objective_function(model, data)

#setting 1: some snapshots for victim, some snapshots for attacker's shadow

def run():
    print('start')
    args = parser.parse_args()
    datanames = ['DBLP5', 'DBLP3', 'reddit', 'Brain']
    attack_model_types = ['DCRNN', 'GConvGRU', 'TGCN', 'A3TGCN', 'GCN']
    victim_types = ['DCRNN', 'GConvGRU', 'TGCN', 'A3TGCN']
    shadow_types = ['DCRNN', 'GConvGRU', 'TGCN', 'A3TGCN']
    url = 'features.txt'
    url2 = 'features2.txt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion_bce = torch.nn.BCELoss(reduction='mean')
    criterion_mse = torch.nn.MSELoss(reduction='mean')
    criterion_f1 = f1_score
    criterions = {'mse': criterion_mse, 'f1': criterion_f1, 'cross_entropy': criterion_cross_entropy,
                  'bce': criterion_bce}
    #victim_type = 'DCRNN' #DCRNN, XXXEVOLVEGCNO,GConvGRU, TGCN, A3TGCN
    dataname = 'DBLP5' #DBLP5,DBLP3,reddit, Brain, epinion
    attack_model_type = 'TGCN' #STG #DCRNN, GConvGRU, TGCN, A3TGCN, GCN
    defence_type = 'DP'  # raw, relaxloss, DP,adver
    victim_type1 = victim_types[0]
    victim_type2 = victim_types[1]
    victim_type3 = victim_types[2]
    victim_type4 = victim_types[3]

    shadow_type1 = shadow_types[0]
    shadow_type2 = shadow_types[1]
    shadow_type3 = shadow_types[2]
    shadow_type4 = shadow_types[3]

    attack_type1 = str(victim_types[0])+'-'+str(shadow_types[0])
    attack_type2 = str(victim_types[1]) + '-' + str(shadow_types[1])
    attack_type3 = str(victim_types[2]) + '-' + str(shadow_types[2])
    attack_type4 = str(victim_types[3]) + '-' + str(shadow_types[3])
    loader = DBLPLoader(dataname)
    print('dataset is loaded')
    dataset = loader.get_dataset()
    #node_features = torch.tensor(dataset.features).shape[2]
    num_classes = torch.tensor(dataset.targets).shape[2]
    num_node = torch.tensor(dataset.targets).shape[1]
    #ratio = 0.6
    #victim_loader, attack_loader = temporal_signal_split(dataset, ratio)
    #victim_division_ratio = 0.5
    #attack_division_ratio = 0.5
    train_test_ratio = 0.7
    victim_shadow_ratio = 0.5
    shadow_loader, victim_loader = temporal_signal_split(dataset, victim_shadow_ratio)
    #victim_loader1, victim_loader2 = temporal_signal_split(victim_loader, victim_division_ratio)
    #attack_loader1, attack_loader2 = temporal_signal_split(attack_loader, attack_division_ratio)
    if dataname =='DBLP5':
        victim_lr = 0.015
        shadow_lr = 0.015
        attack_lr = 0.005
    elif dataname== 'DBLP3':
        victim_lr = 0.015
        shadow_lr = 0.015
        attack_lr = 0.005
    elif dataname == 'reddit':
        victim_lr = 0.015
        shadow_lr = 0.015
        attack_lr = 0.005
    elif dataname == 'Brain':
        victim_lr = 0.015
        shadow_lr = 0.015
        attack_lr = 0.005
    elif dataname == 'epinion':
        victim_lr = 0.1
        shadow_lr = 0.1
        attack_lr = 0.1

    #dataset = None
    #features = None
    #print(victim_type)
    print('Loading victim_model')
    if defence_type =='raw':
        new_victim_model1 = raw_victim_model(args, dataname, victim_type1, victim_loader, train_test_ratio, victim_lr,
                                             device)
        new_victim_model2 = raw_victim_model(args, dataname, victim_type2, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model3 = raw_victim_model(args, dataname, victim_type3, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model4 = raw_victim_model(args, dataname, victim_type4, victim_loader, train_test_ratio, victim_lr,
                                            device)
    elif defence_type == 'relaxloss':
        new_victim_model1 = relax_victim_model(args, dataname, victim_type1, victim_loader, train_test_ratio, victim_lr,
                                             device)
        new_victim_model2 = relax_victim_model(args, dataname, victim_type2, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model3 = relax_victim_model(args, dataname, victim_type3, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model4 = relax_victim_model(args, dataname, victim_type4, victim_loader, train_test_ratio, victim_lr,
                                            device)
    elif defence_type == 'adver':
        new_victim_model1 = adver_victim_model(args, dataname, victim_type1, victim_loader, train_test_ratio, victim_lr,
                                             device)
        new_victim_model2 = adver_victim_model(args, dataname, victim_type2, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model3 = adver_victim_model(args, dataname, victim_type3, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model4 = adver_victim_model(args, dataname, victim_type4, victim_loader, train_test_ratio, victim_lr,
                                            device)
    elif defence_type == 'DP':
        new_victim_model1 = DP_victim_model(args, dataname, victim_type1, victim_loader, train_test_ratio, victim_lr,
                                             device)
        new_victim_model2 = DP_victim_model(args, dataname, victim_type2, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model3 = DP_victim_model(args, dataname, victim_type3, victim_loader, train_test_ratio, victim_lr,
                                               device)
        new_victim_model4 = DP_victim_model(args, dataname, victim_type4, victim_loader, train_test_ratio, victim_lr,
                                            device)



    #objective_function(new_victim_model1, victim_loader, round(num_node * train_test_ratio), device, type='train')
    #objective_function(new_victim_model1, victim_loader, round(num_node * train_test_ratio), device, type='test')

    print('Loading shadow_model')
    #shadow_model = DBLP5_shadow_model(args, shadow_type, victim_loader1, device)
    new_shadow_models1 = raw_shadow_model(args, dataname, shadow_type1, shadow_loader, train_test_ratio, shadow_lr, device, new_victim_model1)
    new_shadow_models2 = raw_shadow_model(args, dataname, shadow_type2, shadow_loader, train_test_ratio, shadow_lr,
                                         device, new_victim_model2)
    new_shadow_models3 = raw_shadow_model(args, dataname, shadow_type3, shadow_loader, train_test_ratio, shadow_lr,
                                         device, new_victim_model3)
    new_shadow_models4 = raw_shadow_model(args, dataname, shadow_type4, shadow_loader, train_test_ratio, shadow_lr,
                                         device, new_victim_model4)
    node_features = torch.tensor(dataset.targets).shape[2]
    print('Loading attack_model')
    new_attack_model1 = raw_attack_model(attack_model_type, defence_type, args, dataname, victim_loader, shadow_loader, new_victim_model1,
                                         new_shadow_models1, node_features, num_classes, attack_type1, device) #获取多个shadow attack
    new_attack_model1.fit(shadow_loader, train_test_ratio, num_epoches=300, learning_rate=attack_lr,
                         criterion=criterions['bce'].to(device))
    new_attack_model2 = raw_attack_model(attack_model_type, defence_type, args, dataname, victim_loader, shadow_loader,
                                        new_victim_model2, new_shadow_models2, node_features, num_classes, attack_type2,
                                        device)
    new_attack_model2.fit(shadow_loader, train_test_ratio, num_epoches=300, learning_rate=attack_lr,
                         criterion=criterions['bce'].to(device))
    new_attack_model3 = raw_attack_model(attack_model_type, defence_type, args, dataname, victim_loader, shadow_loader,
                                        new_victim_model3, new_shadow_models3, node_features, num_classes, attack_type3,
                                        device)
    new_attack_model3.fit(shadow_loader, train_test_ratio, num_epoches=300, learning_rate=attack_lr,
                         criterion=criterions['bce'].to(device))
    new_attack_model4 = raw_attack_model(attack_model_type, defence_type, args, dataname, victim_loader, shadow_loader,
                                        new_victim_model4, new_shadow_models4, node_features, num_classes, attack_type4,
                                        device)
    new_attack_model4.fit(shadow_loader, train_test_ratio, num_epoches=300, learning_rate=attack_lr,
                         criterion=criterions['bce'].to(device))

    #attack_model.fit(victim_loader1, victim_loader2, num_epoches=1000, learning_rate=0.01,  criterion=criterions['bce'].to(device))
    #DBLP5:0.005 #DBLP3:0.03 #reddit, Brain:0.08
    #new_attack_model.fit(shadow_loader, train_test_ratio, num_epoches=500, learning_rate=attack_lr, criterion=criterions['bce'].to(device))
    print('attack_model testing')
    #exit()
    result1, accuracy1 = new_attack_model1.infer(victim_loader, train_test_ratio)
    result2, accuracy2 = new_attack_model2.infer(victim_loader, train_test_ratio)
    result3, accuracy3 = new_attack_model3.infer(victim_loader, train_test_ratio)
    result4, accuracy4 = new_attack_model4.infer(victim_loader, train_test_ratio)

    a1 = float(objective_function(new_victim_model1, victim_loader, round(num_node * train_test_ratio), device, type=None))
    a2 = float(objective_function(new_victim_model2, victim_loader, round(num_node * train_test_ratio), device, type=None))
    a3 = float(objective_function(new_victim_model3, victim_loader, round(num_node * train_test_ratio), device, type=None))
    a4 = float(objective_function(new_victim_model4, victim_loader, round(num_node * train_test_ratio), device, type=None))
    print('F1-SCORE')
    for r in [a1/result1,a2/result2,a3/result3,a4/result4]:
        print(r)
    print('Accuracy')
    for a in [a1/accuracy1,a2/accuracy2,a3/accuracy3,a4/accuracy4]:
        print(a)



    #a = objective_function(new_victim_model, victim_loader, round(num_node * train_test_ratio), device, type=None)
    #print('f1-score trade-off is '+str(a/result))
    #print('accuracy trade-off is '+str(a/accuracy))
    #objective_function(model, victim_loader, split, device, type='train')
    #objective_function(model, victim_loader, split, device, type='test')
    #new_attack_model.infer(attack_loader, attack_division_ratio)
    #attack_model.infer(attack_loader, attack_division_ratio, type='precision')
    #attack_model.infer(attack_loader, attack_division_ratio, type='recall')


run()