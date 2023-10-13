from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def load_arxiv(per_classnum=10, seed=0, thred=1.0, lam=-1, model_type=''):
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root='./dataset/')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    edge_index_2 = torch.zeros_like(data.edge_index)
    edge_index_2[0] = data.edge_index[1]
    edge_index_2[1] = data.edge_index[0]
    data.edge_index = torch.cat([data.edge_index, edge_index_2], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    
    # split data
    node_id = [[] for _ in range(data.y.max()+1)]
    train_id = np.arange(data.num_nodes)[train_mask]
    for i in range(train_id.shape[0]):
        node_id[data.y[train_id[i]]].append(train_id[i])
    few_shot_train_id = []
    for i in range(data.y.max()+1):
        np.random.shuffle(node_id[i])
        few_shot_train_id = few_shot_train_id + node_id[i][:per_classnum]

    data.y = data.y.squeeze(1)
    
    new_embed = np.load('./dataset/ogbn_arxiv_process/new_sample.npy')
    text_embed = np.load('./dataset/ogbn_arxiv_process/raw_text.npy')

    text_embed = np.concatenate((text_embed, new_embed))
    N = data.num_nodes
    M = int(new_embed.shape[0])
    C = data.y.max() + 1
    K = int(new_embed.shape[0] / C)
    new_y = []
    for i in range(C):
        new_y = new_y + [i]*K
    data.y = torch.cat((data.y, torch.tensor(new_y)))

    if model_type == 'Edge':
        E = data.edge_index.size(1)
        print('Exist edge:', E)
        # random sample negative edges
        missing_edge_index = torch.load('./dataset/ogbn_arxiv_process/neg_edge_idx.pt')
        data.edge_y = torch.cat([torch.ones(E), torch.zeros(E)])
        data.train_edge = torch.cat([data.edge_index, missing_edge_index], dim=1)
    
        # Filtering predicting pairs
        sim = torch.mm(torch.from_numpy(new_embed), torch.from_numpy(text_embed).t())
        new_edge = torch.nonzero(sim > thred)
        new_edge[:,0] = new_edge[:,0] + N
        data.predict_edge_index = new_edge.transpose(0,1)
        print('predict size:', data.predict_edge_index.size())
    elif model_type == 'Node':
        few_shot_train_id = few_shot_train_id + list(range(N, N+M))
        data.num_nodes = N + M
        data.train_mask = torch.tensor(
            [x in few_shot_train_id for x in range(data.num_nodes)])
        data.val_mask = torch.cat((data.val_mask, torch.zeros(M).bool()))
        data.test_mask = torch.cat((data.test_mask, torch.zeros(M).bool()))
        if lam != -1:
            data.edge_index = torch.load('../../dataset/ogbn_arxiv_process/adj/lam_' + str(lam) + '_thred_' + str(thred) + '.pt')
    else:
        print('error')
        exit()
    return data, text_embed