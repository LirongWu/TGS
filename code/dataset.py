import dgl
import torch
import numpy as np

def dataloader(param, device, mode):
    
    if param['dataset'] == 'cora':
        graph = dgl.data.CoraGraphDataset()[0]
    elif param['dataset'] == 'citeseer':
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif param['dataset'] == 'coauthor-cs':
        graph = dgl.data.CoauthorCSDataset()[0]
    elif param['dataset'] == 'coauthor-phy':
        graph = dgl.data.CoauthorPhysicsDataset()[0] 
    elif param['dataset'] == 'amazon-com':
        graph = dgl.data.AmazonCoBuyComputerDataset()[0]
    elif param['dataset'] == 'amazon-photo':
        graph = dgl.data.AmazonCoBuyPhotoDataset()[0] 


    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    if param['dataset'] == 'cora' or param['dataset'] == 'citeseer':
        train_mask = graph.ndata['train_mask'].to(device)
        val_mask = graph.ndata['val_mask'].to(device)
        test_mask = graph.ndata['test_mask'].to(device)
    else:

        train_index = torch.tensor(np.load("../dataset/{}/train_mask.npy".format(param['dataset']))).to(device)
        val_index = torch.tensor(np.load("../dataset/{}/val_mask.npy".format(param['dataset']))).to(device)
        test_index = torch.tensor(np.load("../dataset/{}/test_mask.npy".format(param['dataset']))).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        val_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        test_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

    if mode > 0 and mode < 10:
        train_index = torch.tensor(np.load("../dataset/{}/train_mask_r{}.npy".format(param['dataset'], mode))).to(device)
        val_index = torch.tensor(np.load("../dataset/{}/val_mask_r{}.npy".format(param['dataset'], mode))).to(device)
        test_index = torch.tensor(np.load("../dataset/{}/test_mask_r{}.npy".format(param['dataset'], mode))).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        val_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        test_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

    if mode > 10 and mode < 20:
        labels = torch.LongTensor(np.load("../dataset/{}/label_mask_r{}.npy".format(param['dataset'], mode))).to(device)

    return graph, features, labels, train_mask, val_mask, test_mask