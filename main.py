import torch
import numpy as np
import networkx as nx
from dataloader import DBLPLoader
import argparse
from victim import raw_victim_model, relax_victim_model, adver_victim_model, DP_victim_model, \
    LapDP_victim_model, STSA_victim_model, DP_STSA_victim_model, objective_function, objective_function2
from attack import raw_attack_model
from shadow import raw_shadow_model
from sklearn.metrics import f1_score
from torch_geometric_temporal.signal import temporal_signal_split,train_test_split

#from TGCN.signal.train_test_split import temporal_signal_split
import os

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")

parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name')
parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    # Number of negative samples per positive pair.
parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='64,32',
                        help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='8',
                        help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
parser.add_argument('--epsilon', type=float, default=1.0,
                    help='privacy budget for DP methods, e.g., 0.5, 1, 2')

#def objective_function(model, data)

#setting 1: some snapshots for victim, some snapshots for attacker's shadow

def main():
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

    attack_model_type = 'GCN' #STG #DCRNN, GConvGRU, TGCN, A3TGCN, GCN
    defence_type = 'DP-STSA'
    # raw, relaxloss, adver, GauDP, LapDP, STSA, DP-STSA
    victim_type = 'GConvGRU'
    shadow_type = 'GConvGRU'
    attack_type = str(victim_type)+'-'+str(shadow_type)
    loader = DBLPLoader(dataname)
    print('dataset is loaded')
    dataset = loader.get_dataset()
    #node_features = torch.tensor(dataset.features).shape[2]
    num_classes = torch.tensor(dataset.targets).shape[2]
    num_node = torch.tensor(dataset.targets).shape[1]
    train_test_ratio = 0.7
    victim_shadow_ratio = 0.5
    shadow_loader, victim_loader = temporal_signal_split(dataset, victim_shadow_ratio)
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

    print('Loading victim_model')


    #new_victim_model1 = raw_victim_model(
    #    args, dataname, victim_type, victim_loader,
    #    train_test_ratio, victim_lr, device
    #)

    #new_victim_model2 = relax_victim_model(
    #    args, dataname, victim_type, victim_loader,
    #    train_test_ratio, victim_lr, device
    #)

    #new_victim_model3 = adver_victim_model(
    #    args, dataname, victim_type, victim_loader,
    #    train_test_ratio, victim_lr, device
    #)

    new_victim_model4 = DP_victim_model(
        args, dataname, victim_type, victim_loader,
        train_test_ratio, victim_lr, device
    )

    new_victim_model5 = LapDP_victim_model(
        args, dataname, victim_type, victim_loader,
        train_test_ratio, victim_lr, device
    )
    new_victim_model6 = STSA_victim_model(
        args, dataname, victim_type, victim_loader,
        train_test_ratio, victim_lr, device
    )

    new_victim_model7 = DP_STSA_victim_model(
        args, dataname, victim_type, victim_loader,
        train_test_ratio, victim_lr, device
    )
    
    print('Loading shadow_model')
    #shadow_model = DBLP5_shadow_model(args, shadow_type, victim_loader1, device)
    new_shadow_models = raw_shadow_model(args, dataname, shadow_type, shadow_loader, train_test_ratio, shadow_lr, device, new_victim_model)
    node_features = torch.tensor(dataset.targets).shape[2]
    print('Loading attack_model')
    new_attack_model1 = raw_attack_model(
        attack_model_type, 'raw', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model1.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model2 = raw_attack_model(
        attack_model_type, 'relaxloss', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model2.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model3 = raw_attack_model(
        attack_model_type, 'adver', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model3.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model4 = raw_attack_model(
        attack_model_type, 'GauDP', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model4.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model5 = raw_attack_model(
        attack_model_type, 'LapDP', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model5.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model6 = raw_attack_model(
        attack_model_type, 'STSA', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model6.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    new_attack_model7 = raw_attack_model(
        attack_model_type, 'DP-STSA', args, dataname,
        victim_loader, shadow_loader, new_victim_model,
        new_shadow_models, node_features, num_classes,
        attack_type, device
    )
    new_attack_model7.fit(
        shadow_loader, train_test_ratio, num_epoches=300,
        learning_rate=attack_lr, criterion=criterions['bce'].to(device)
    )

    if defence_type == 'raw':
        new_attack_model = new_attack_model1
    elif defence_type == 'relaxloss':
        new_attack_model = new_attack_model2
    elif defence_type == 'adver':
        new_attack_model = new_attack_model3
    elif defence_type == 'GauDP':
        new_attack_model = new_attack_model4
    elif defence_type == 'LapDP':
        new_attack_model = new_attack_model5
    elif defence_type == 'STSA':
        new_attack_model = new_attack_model6
    elif defence_type == 'DP-STSA':
        new_attack_model = new_attack_model7
    else:
        raise ValueError('Unknown defence_type: {}'.format(defence_type))

    #attack_model.fit(victim_loader1, victim_loader2, num_epoches=1000, learning_rate=0.01,  criterion=criterions['bce'].to(device))
    #DBLP5:0.005 #DBLP3:0.03 #reddit, Brain:0.08
    #new_attack_model.fit(shadow_loader, train_test_ratio, num_epoches=500, learning_rate=attack_lr, criterion=criterions['bce'].to(device))
    print('attack_model testing')
    #exit()

    if defence_type in ['STSA', 'DP-STSA']:
        result, accuracy = new_attack_model.infer2(
            victim_loader,
            train_test_ratio
        )
        a = objective_function2(
            new_victim_model,
            victim_loader,
            round(num_node * train_test_ratio),
            device,
            type=None
        )
    else:
        result, accuracy = new_attack_model.infer(
            victim_loader,
            train_test_ratio
        )
        a = objective_function(
            new_victim_model,
            victim_loader,
            round(num_node * train_test_ratio),
            device,
            type=None
        )

    print('f1-score trade-off is '+str(a/result))
    print('accuracy trade-off is '+str(a/accuracy))
    #objective_function(model, victim_loader, split, device, type='train')
    #objective_function(model, victim_loader, split, device, type='test')
    #new_attack_model.infer(attack_loader, attack_division_ratio)
    #attack_model.infer(attack_loader, attack_division_ratio, type='precision')
    #attack_model.infer(attack_loader, attack_division_ratio, type='recall')

if __name__=='__main__':
    main()

