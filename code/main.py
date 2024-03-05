import os
import csv
import nni
import time
import json
import warnings
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import *
from model import *
from utils import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    model = MLP(features.shape[1], param['hid_dim'], n_class, param).to(device)
    factor = NeFactor(features.shape[1], param['hid_dim'], param).to(device)

    optimize_model = torch.optim.Adam(list(model.parameters()), lr=float(param['lr']), weight_decay=float(param['weight_decay']))
    optimize_factor = torch.optim.Adam(list(factor.parameters()), lr=1e-4, weight_decay=float(param['weight_decay']))

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(param['epochs']):

        neg = torch.randint(0, n_nodes, (n_edges, ))

        for idx in DataLoader(range(n_edges), batch_size=param['batch_size'], shuffle=True):
            
            model.train()

            neg_idx = neg[idx]
            src_idx = src[idx]
            dst_idx = dst[idx]

            # # Learning interpolation coefficients
            beta_src = factor(features[src_idx], features[dst_idx])
            beta_dst = factor(features[dst_idx], features[src_idx])

            y, z = model(features[neg_idx])
            y1, z1 = model(features[src_idx])
            y2, z2 = model(features[dst_idx])
            y3, z3 = model(features[dst_idx] * beta_dst + features[src_idx] * (1-beta_dst))
            y4, z4 = model(features[src_idx] * beta_src + features[dst_idx] * (1-beta_src))

            # Neighborhood Self-Distillation
            loss_ne = F.mse_loss(y1, z3) + F.mse_loss(y2, z4) - F.mse_loss(torch.softmax(y1, dim=-1), torch.softmax(z, dim=-1)) - F.mse_loss(torch.softmax(y2, dim=-1), torch.softmax(z, dim=-1))

            loss_cla = torch.zeros((1)).to(device)
            m = train_mask[src_idx]
            if m.any().item():
                target = labels[src_idx][m]
                loss_cla += F.cross_entropy(y1[m], target, weight=weight) + F.cross_entropy(z3[m], target, weight=weight)

            m = train_mask[dst_idx]
            if m.any().item():
                target = labels[dst_idx][m]
                loss_cla += F.cross_entropy(y2[m], target, weight=weight) + F.cross_entropy(z4[m], target, weight=weight)

            loss_total = loss_ne + param['alpha'] * loss_cla
            optimize_model.zero_grad()
            optimize_factor.zero_grad()
            loss_total.backward()
            optimize_model.step()
            optimize_factor.step()

        # Model Inferring
        model.eval()
        logits, _ = model(features)

        train_acc = ((logits[train_mask].max(dim=1).indices == labels[train_mask]).sum() / train_mask.sum().float()).item()
        val_acc = ((logits[val_mask].max(dim=1).indices == labels[val_mask]).sum() / val_mask.sum().float()).item()
        test_acc = ((logits[test_mask].max(dim=1).indices == labels[test_mask]).sum() / test_mask.sum().float()).item()

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if es >= 50:
                print("Early stopping!")
                break

        print("\033[0;30;46m [{}] Ne: {:.5f}, Cla: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                    epoch, loss_ne.item(), loss_cla.item(), loss_total.item(), train_acc, val_acc, test_acc, val_best, test_val, test_best))
     
    return test_acc, test_val, test_best


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'coauthor-cs', 'coauthor-phy', 'amazon-com', 'amazon-photo'])
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_mode', type=int, default=0)
    parser.add_argument('--data_mode', type=int, default=0)
    parser.add_argument('--model_mode', type=int, default=0)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    elif param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    elif param['data_mode'] == 2:
        param['dataset'] = 'coauthor-cs'
    elif param['data_mode'] == 3:
        param['dataset'] = 'coauthor-phy'
    elif param['data_mode'] == 4:
        param['dataset'] = 'amazon-com'
    elif param['data_mode'] == 5:
        param['dataset'] = 'amazon-photo'

    if os.path.exists("../param/best_parameters.json"):
        param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']]

    graph, features, labels, train_mask, val_mask, test_mask = dataloader(param, device, param['model_mode'])

    src, dst = graph.edges()
    n_nodes = features.shape[0]
    n_class = int(labels.max().item() + 1)
    n_edges = src.shape[0]

    label_ndist = labels[torch.arange(n_nodes)[train_mask]].float().histc(n_class)
    label_edist = (labels[src[train_mask[src]]].float().histc(n_class) + labels[dst[train_mask[dst]]].float().histc(n_class))
    weight = n_class * F.normalize(label_ndist / label_edist, p=1, dim=0)

    if param['save_mode'] == 0:
        SetSeed(param['seed'])
        test_acc, test_val, test_best = main()
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []

        for seed in range(5):
            SetSeed(seed + param['seed'] * 5)
            test_acc, test_val, test_best = main()
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            nni.report_intermediate_result(test_val)
        nni.report_final_result(np.mean(test_val_list))

    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    
    if param['save_mode'] == 0:
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))

    else:  
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
    writer.writerow(results)