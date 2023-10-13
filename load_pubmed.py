import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd

dir = './dataset'

def get_pubmed_casestudy(corrected=False, SEED=0):
    data_Y, data_edges = parse_pubmed()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    dataset = Planetoid(dir, data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)
    data.num_nodes = len(data_Y)

    return data


def parse_pubmed():
    path = dir + '/PubMed_orig/data/'

    n_nodes = 19717

    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}

    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        node_file.readline()
        node_file.readline()
        
        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            label = int(items[1].split('=')[-1]) - 1  
            data_Y[i] = label

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_Y, np.unique(data_edges, axis=0).transpose()


def load_pubmed(per_classnum=10, seed=0, thred=1.0, lam=-1, model_type=''):
    data = get_pubmed_casestudy(False, seed)
    np.random.seed(seed)
    # split data
    node_id = [[] for _ in range(data.y.max()+1)]
    for i in range(len(data.y)):
        node_id[data.y[i]].append(i)
    train_id = []
    remain_id = []
    for i in range(data.y.max()+1):
        np.random.shuffle(node_id[i])
        train_id = train_id + node_id[i][:per_classnum]
        remain_id = remain_id + node_id[i][per_classnum:]
    
    np.random.shuffle(remain_id)
    data.train_id = train_id
    data.val_id = remain_id[0:500]
    data.test_id = remain_id[500:1500]

    new_embed = np.load(dir + '/pubmed_process/pubmed_new_sample.npy')
    text_embed = np.load(dir + '/pubmed_process/pubmed_text_embed.npy')
    text_embed = np.concatenate((text_embed, new_embed))
    N = data.num_nodes
    M = int(new_embed.shape[0])
    C = data.y.max() + 1
    K = int(new_embed.shape[0] / C)
    new_y = []
    for i in range(C):
        new_y = new_y + [i]*K
    data.y = torch.cat((data.y, torch.tensor(new_y)))
    # data.x 目前不使用
    data.train_id = data.train_id + list(range(N, N+M))
    data.num_nodes = N + M
    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    
    if model_type == 'Edge':
        E = data.edge_index.size(1)
        sparse_matrix = torch.sparse.FloatTensor(data.edge_index,
            torch.ones(data.edge_index.size(1)),
            torch.Size([N, N]))
        adj = sparse_matrix.to_dense()
        missing_edge = torch.nonzero(adj != 1)
        missing_edge_idx = np.arange(0, missing_edge.size(0))
        np.random.shuffle(missing_edge_idx)
        missing_edge_index = missing_edge[missing_edge_idx[0:E]].transpose(0, 1)
        data.edge_y = torch.cat([torch.ones(E), torch.zeros(E)])
        data.train_edge = torch.cat([data.edge_index, missing_edge_index], dim=1)
    
        sim = torch.mm(torch.from_numpy(new_embed), torch.from_numpy(text_embed).t())
        new_edge = torch.nonzero(sim > thred)
        new_edge[:,0] = new_edge[:,0] + N
        data.predict_edge_index = new_edge.transpose(0,1)
        print('predict size:', data.predict_edge_index.size())
    elif model_type == 'Node':
        if lam == -1:
            pass
        else:
            data.edge_index = torch.load('./dataset/pubmed_process/adj/lam_' + str(lam) + '_thred_' + str(thred) + '.pt')
    else:
        print('error')
        exit()
    return data, text_embed
