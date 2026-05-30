import torch
import numpy as np
from models import  DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN, GConvGRU_RecurrentGCN, Robust_RGNN
from tqdm import tqdm
import torch.nn.functional as F
from deeprobust.graph.global_attack import DICE
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import torch_geometric
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import copy as cp
import os
from pyvacy import optim

from torch.optim import Optimizer

def add_gaussian_noise_to_gradients(model, epsilon=1.0, delta=1e-5, l2_norm_clip=1.0):
    """
    Add Gaussian DP noise to model gradients.

    This function is used by DP_victim_model as GauDP baseline.
    It performs:
    1. Global L2 gradient clipping.
    2. Gaussian noise injection.

    Gaussian noise scale:
        sigma = sqrt(2 * log(1.25 / delta)) * C / epsilon
    where C is l2_norm_clip.
    """

    epsilon = float(epsilon)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return

    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), p=2)
            for p in parameters
        ]),
        p=2
    )

    clip_coef = l2_norm_clip / (total_norm + 1e-6)
    clip_coef = min(clip_coef.item(), 1.0)

    noise_scale = np.sqrt(2.0 * np.log(1.25 / delta)) * l2_norm_clip / epsilon

    for p in parameters:
        p.grad.detach().mul_(clip_coef)

        noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=p.grad.shape,
            device=p.grad.device
        )

        p.grad.add_(noise)
def gaussian_noise_multiplier_from_epsilon(epsilon, delta=1e-5):
    """
    Convert epsilon to Gaussian noise multiplier.

    This is used by GauDP baseline.
    A smaller epsilon produces a larger Gaussian noise multiplier.
    """
    epsilon = float(epsilon)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    return np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def laplace_noise_scale_from_epsilon(epsilon, l2_norm_clip=1.0):
    """
    Compute Laplace noise scale.

    This is used by LapDP baseline.
    The scale is C / epsilon, where C is the clipping bound.
    """
    epsilon = float(epsilon)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    return float(l2_norm_clip) / epsilon

def add_laplace_noise_to_stsa_gradients(
        model,
        epsilon=1.0,
        Cs=5.0,
        Ct=5.0,
        tau=1.0
):
    """
    Add Laplace DP noise to STSA gradients.

    DP-STSA:
    - spatial / structural gradients use clipping bound Cs
    - temporal gradients use clipping bound Ct
    - privacy budget is split by tau

    epsilon_s = epsilon / (1 + tau)
    epsilon_t = epsilon * tau / (1 + tau)
    """

    spatial_params = []
    temporal_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        lname = name.lower()

        if ('structural' in lname) or ('spatial' in lname) or ('structure' in lname):
            spatial_params.append(param)
        elif ('temporal' in lname) or ('time' in lname):
            temporal_params.append(param)
        else:
            other_params.append(param)

    # 如果 Robust_RGNN 的参数名里没有 spatial / temporal 关键词，
    # 防止所有参数都进 other_params 后完全不加噪。
    # 默认把未识别参数归为空间结构参数。
    if len(spatial_params) == 0 and len(temporal_params) == 0:
        spatial_params = other_params
        other_params = []

    epsilon_s = epsilon / (1.0 + tau)
    epsilon_t = epsilon * tau / (1.0 + tau)

    epsilon_s = max(epsilon_s, 1e-12)
    epsilon_t = max(epsilon_t, 1e-12)

    def clip_and_add_laplace_noise(params, clip_bound, eps_part):
        if len(params) == 0:
            return

        valid_params = [p for p in params if p.grad is not None]

        if len(valid_params) == 0:
            return

        device = valid_params[0].grad.device

        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), p=2).to(device)
                for p in valid_params
            ]),
            p=2
        )

        clip_coef = clip_bound / (total_norm + 1e-6)
        clip_coef = min(clip_coef.item(), 1.0)

        # Laplace mechanism: scale = sensitivity / epsilon
        noise_scale = clip_bound / eps_part

        for p in valid_params:
            p.grad.data.mul_(clip_coef)

            laplace_dist = torch.distributions.Laplace(
                loc=torch.tensor(0.0, device=p.grad.device),
                scale=torch.tensor(noise_scale, device=p.grad.device)
            )

            noise = laplace_dist.sample(p.grad.shape)

            p.grad.data.add_(noise)

    clip_and_add_laplace_noise(spatial_params, Cs, epsilon_s)
    clip_and_add_laplace_noise(temporal_params, Ct, epsilon_t)

    # 未识别参数默认按照 spatial 部分处理
    if len(other_params) > 0:
        clip_and_add_laplace_noise(other_params, Cs, epsilon_s)

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
    """
    GauDP baseline.

    This function keeps the same input and output format as other victim methods.
    It trains the victim model with Gaussian DP gradient perturbation.

    It does not depend on pyvacy.optim.DPSGD, because different pyvacy versions
    have incompatible DPSGD arguments.
    """

    epsilon = getattr(args, 'epsilon', 1.0)
    delta = getattr(args, 'delta', 1e-5)
    l2_norm_clip = getattr(args, 'l2_norm_clip', 1.0)
    epochs = getattr(args, 'epochs', 500)

    url = (
        str(dataname)
        + '/victim/'
        + str(victim_type)
        + '-GauDP-eps'
        + str(epsilon)
    )

    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)

    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    else:
        raise ValueError("Unsupported victim_type: {}".format(victim_type))

    model = model.to(device)

    if os.path.exists(url) is False:
        file = open(url, 'w')
        file.close()

    if os.path.getsize(url) > 0:
        print('Saved GauDP victim model is loaded')
        weights = torch.load(f=url, map_location=device)
        model.load_state_dict(weights['victim_model'], strict=False)
        return model

    print('The GauDP victim model is loaded')
    print('epsilon:', epsilon)
    print('delta:', delta)
    print('l2_norm_clip:', l2_norm_clip)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    model.train()

    for epoch in tqdm(range(epochs)):
        Hidden = None
        cost = 0

        optimizer.zero_grad()

        for time, snapshot in enumerate(victim_loader):
            y = snapshot.y

            if isinstance(y, torch.Tensor):
                labels = torch.argmax(y, dim=1).long().to(device)
            else:
                y = np.argmax(y, axis=1)
                labels = torch.from_numpy(y).long().to(device)

            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)

            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)

            cost = cost + criterion(y_hat[:split], labels[:split])

        cost = cost / (time + 1)

        print('The ' + str(epoch) + ' training loss is ' + str(cost))

        cost.backward()

        add_gaussian_noise_to_gradients(
            model=model,
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip
        )

        optimizer.step()

    model.eval()

    torch.save({'victim_model': model.state_dict()}, f=url)

    objective_function(
        model,
        victim_loader,
        split,
        device,
        type='train'
    )

    objective_function(
        model,
        victim_loader,
        split,
        device,
        type='test'
    )

    return model

def LapDP_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    """
    LapDP baseline.

    This function keeps the same input and output format as other victim methods.
    It trains the victim model with Laplace DPSGD.
    """

    epsilon = getattr(args, 'epsilon', 1.0)
    l2_norm_clip = getattr(args, 'l2_norm_clip', 1.0)
    epochs = getattr(args, 'epochs', 500)

    url = (
        str(dataname)
        + '/victim/'
        + str(victim_type)
        + '-LapDP-eps'
        + str(epsilon)
    )

    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)

    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'GConvGRU':
        model = GConvGRU_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(
            node_features=node_features,
            num_classes=num_classes
        )
    else:
        raise ValueError("Unsupported victim_type: {}".format(victim_type))

    model = model.to(device)

    if os.path.exists(url) is False:
        file = open(url, 'w')
        file.close()

    if os.path.getsize(url) > 0:
        print('Saved LapDP victim model is loaded')
        weights = torch.load(f=url, map_location=device)
        model.load_state_dict(weights['victim_model'], strict=False)
        return model

    print('The LapDP victim model is loaded')
    print('epsilon:', epsilon)
    print('l2_norm_clip:', l2_norm_clip)

    optimizer = LaplaceDPSGD(
        params=model.parameters(),
        lr=lr,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        weight_decay=0.0
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    model.train()

    for epoch in tqdm(range(epochs)):
        Hidden = None
        cost = 0

        optimizer.zero_grad()

        for time, snapshot in enumerate(victim_loader):
            optimizer.zero_microbatch_grad()

            y = snapshot.y

            if isinstance(y, torch.Tensor):
                labels = torch.argmax(y, dim=1).long().to(device)
            else:
                y = np.argmax(y, axis=1)
                labels = torch.from_numpy(y).long().to(device)

            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device)

            y_hat, Hidden = model(x, edge_index, edge_weight, Hidden)

            cost = criterion(y_hat[:split], labels[:split])
            cost.backward()

            optimizer.microbatch_step()

        print('The ' + str(epoch) + ' training loss is ' + str(cost))

        optimizer.step()

    model.eval()

    torch.save({'victim_model': model.state_dict()}, f=url)

    objective_function(model, victim_loader, split, device, type='train')
    objective_function(model, victim_loader, split, device, type='test')

    return model

def STSA_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    #url = str(dataname)+'/victim/'+str(victim_type)+'-STSA'
    url = str(dataname) + '/victim/' + str(victim_type) + '-STSA3'
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

def add_noise_to_stsa_gradients(
    model,
    epsilon=1.0,
    delta=1e-5,
    Cs=5.0,
    Ct=5.0,
    tau=1.0,
    noise_type="gaussian"
):
    """
    Add DP noise to spatial-temporal self-attention gradients.

    This function is used by DP-STSA.
    It separates spatial and temporal attention parameters,
    allocates privacy budgets according to their gradient sensitivities,
    and adds Gaussian or Laplace noise to the corresponding gradients.
    """

    spatial_params = []
    temporal_params = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        lower_name = name.lower()

        if 'structural' in lower_name or 'spatial' in lower_name:
            spatial_params.append(p)
        elif 'temporal' in lower_name:
            temporal_params.append(p)

    def get_group_norm(params):
        if len(params) == 0:
            return 0.0

        return torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), p=2)
                for p in params
            ]),
            p=2
        ).item()

    spatial_sensitivity = get_group_norm(spatial_params) + 1e-12
    temporal_sensitivity = get_group_norm(temporal_params) + 1e-12

    spatial_weight = spatial_sensitivity ** tau
    temporal_weight = temporal_sensitivity ** tau

    total_weight = spatial_weight + temporal_weight

    if total_weight == 0:
        epsilon_s = epsilon / 2.0
        epsilon_t = epsilon / 2.0
    else:
        epsilon_s = epsilon * temporal_weight / total_weight
        epsilon_t = epsilon * spatial_weight / total_weight

    epsilon_s = max(epsilon_s, 1e-6)
    epsilon_t = max(epsilon_t, 1e-6)

    def clip_and_noise(params, clip_bound, epsilon_group):
        if len(params) == 0:
            return

        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), p=2)
                for p in params
            ]),
            p=2
        )

        clip_coef = min(clip_bound / (total_norm + 1e-6), 1.0)

        for p in params:
            p.grad.detach().mul_(clip_coef)

            if noise_type.lower() == "gaussian":
                noise_scale = np.sqrt(2.0 * np.log(1.25 / delta)) * clip_bound / epsilon_group
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=p.grad.shape,
                    device=p.grad.device
                )
            elif noise_type.lower() == "laplace":
                noise_scale = clip_bound / epsilon_group
                laplace = torch.distributions.Laplace(
                    loc=torch.tensor(0.0, device=p.grad.device),
                    scale=torch.tensor(noise_scale, device=p.grad.device)
                )
                noise = laplace.sample(p.grad.shape)
            else:
                raise ValueError("noise_type must be 'gaussian' or 'laplace'.")

            p.grad.add_(noise)

    clip_and_noise(
        spatial_params,
        clip_bound=Cs,
        epsilon_group=epsilon_s
    )

    clip_and_noise(
        temporal_params,
        clip_bound=Ct,
        epsilon_group=epsilon_t
    )

def DP_STSA_victim_model(args, dataname, victim_type, victim_loader, train_test_ratio, lr, device):
    """
    DP-STSA victim model.

    This version follows STSA_victim_model exactly:
    - same data preprocessing
    - same Robust_RGNN initialization
    - same get_total_loss
    - same objective_function2

    Difference:
    - add Laplace noise to STSA gradients after cost.backward()
    """

    epsilon = getattr(args, 'epsilon', 1.0)
    Cs = getattr(args, 'Cs', 5.0)
    Ct = getattr(args, 'Ct', 5.0)
    tau = getattr(args, 'tau', 1.0)
    epochs = getattr(args, 'epochs', 300)
    eval_every = getattr(args, 'eval_every', 20)

    print('================ DP-STSA Victim Model, Laplace Noise ================')
    print('dataname:', dataname)
    print('victim_type:', victim_type)
    print('epsilon:', epsilon)
    print('Cs:', Cs)
    print('Ct:', Ct)
    print('tau:', tau)
    print('epochs:', epochs)

    url = str(dataname) + '/victim/' + str(victim_type) + '-DP-STSA-Lap-eps' + str(epsilon)

    node_features = torch.tensor(victim_loader.features).shape[2]
    num_classes = torch.tensor(victim_loader.targets).shape[2]
    num_node = torch.tensor(victim_loader.features).shape[1]
    split = round(num_node * train_test_ratio)
    time_length = torch.tensor(victim_loader.features).shape[0]

    model = Robust_RGNN(
        args,
        device,
        node_features,
        time_length,
        num_classes,
        victim_type
    )

    victim_graphs = data_preprossing(victim_loader)

    if torch.cuda.is_available():
        model = model.to(device)

    if os.path.exists(str(dataname) + '/victim') == False:
        os.makedirs(str(dataname) + '/victim')

    if os.path.exists(url) == False:
        file = open(url, 'w')
        file.close()

    if os.path.getsize(url) > 0:
        print('Saved DP-STSA victim model is loaded')
        weights = torch.load(f=url, map_location=device)
        model.load_state_dict(weights['victim_model'], strict=False)
        return model

    print('The data of DP-STSA victim_model is already for loading')
    print('The data of DP-STSA victim_model is loaded')

    optimizer = torch.optim.Adam(model.parameters(), lr)

    time = len(victim_graphs)

    degrees = []
    cens = []
    cen_var = []

    for t in range(time):
        graph = to_networkx(victim_graphs[t])

        degree = torch.tensor(
            list(dict(graph.degree).values()),
            dtype=torch.float
        ).to(device)

        degrees.append(degree)

        cen = torch.tensor(
            list(dict(nx.closeness_centrality(graph)).values()),
            dtype=torch.float
        ).to(device)

        cens.append(cen)

    cen_var.append(torch.zeros(cens[0].shape).to(device))

    for t in range(1, time - 1):
        cen_var.append(
            (abs(cens[t] - cens[t - 1]) + abs(cens[t + 1] - cens[t])).to(device)
        )

    cen_var.append(torch.zeros(cens[time - 1].shape).to(device))

    model.train()

    max_o = 0.0
    best_state_dict = None

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        cost = model.get_total_loss(
            victim_graphs,
            degrees,
            cen_var,
            split
        )

        # 评价当前模型
        if epoch % eval_every == 0:
            o = objective_function2(
                model,
                victim_loader,
                split,
                device,
                type=None
            )

            if isinstance(o, torch.Tensor):
                o_value = o.detach().item()
            else:
                o_value = float(o)

            if o_value > max_o:
                max_o = o_value
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                'DP-STSA-Lap Epoch: {}, Loss: {}, Acc: {}'.format(
                    epoch,
                    cost,
                    o
                )
            )

        cost.backward()

        add_laplace_noise_to_stsa_gradients(
            model,
            epsilon=epsilon,
            Cs=Cs,
            Ct=Ct,
            tau=tau
        )

        optimizer.step()

    model.eval()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(
        {
            'victim_model': model.state_dict()
        },
        f=url
    )

    print('Save DP-STSA-Lap victim model to:', url)
    print('Best DP-STSA-Lap accuracy:', max_o)

    objective_function2(
        model,
        victim_loader,
        split,
        device,
        type='train'
    )

    objective_function2(
        model,
        victim_loader,
        split,
        device,
        type='test'
    )

    return model

class LaplaceDPSGD(Optimizer):
    """
    Laplace DPSGD optimizer for LapDP baseline.

    It performs:
    1. Microbatch gradient clipping.
    2. Gradient accumulation.
    3. Laplace noise injection.
    4. Parameter update.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        l2_norm_clip=1.0,
        epsilon=1.0,
        weight_decay=0.0
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if l2_norm_clip <= 0.0:
            raise ValueError("Invalid l2_norm_clip: {}".format(l2_norm_clip))
        if epsilon <= 0.0:
            raise ValueError("Invalid epsilon: {}".format(epsilon))

        defaults = dict(
            lr=lr,
            l2_norm_clip=l2_norm_clip,
            epsilon=epsilon,
            weight_decay=weight_decay
        )

        super(LaplaceDPSGD, self).__init__(params, defaults)

        self._microbatch_grads = []

        for group in self.param_groups:
            for p in group['params']:
                self._microbatch_grads.append(torch.zeros_like(p.data))

    def zero_microbatch_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def zero_grad(self):
        super(LaplaceDPSGD, self).zero_grad()

        for i in range(len(self._microbatch_grads)):
            self._microbatch_grads[i].zero_()

    def microbatch_step(self):
        total_norm = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5

        l2_norm_clip = self.param_groups[0]['l2_norm_clip']
        clip_coef = min(l2_norm_clip / (total_norm + 1e-6), 1.0)

        i = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self._microbatch_grads[i].add_(p.grad.data * clip_coef)
                i += 1

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        i = 0

        for group in self.param_groups:
            lr = group['lr']
            l2_norm_clip = group['l2_norm_clip']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            noise_scale = laplace_noise_scale_from_epsilon(
                epsilon=epsilon,
                l2_norm_clip=l2_norm_clip
            )

            for p in group['params']:
                if p.requires_grad is False:
                    i += 1
                    continue

                grad = self._microbatch_grads[i]

                noise = torch.distributions.Laplace(
                    loc=torch.tensor(0.0, device=p.device),
                    scale=torch.tensor(noise_scale, device=p.device)
                ).sample(p.data.shape)

                noisy_grad = grad + noise

                if weight_decay != 0:
                    noisy_grad = noisy_grad.add(p.data, alpha=weight_decay)

                p.data.add_(noisy_grad, alpha=-lr)

                i += 1

        return loss

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

