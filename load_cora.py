import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

dir = './dataset'

def get_cora_casestudy(SEED=0):
    data_Y, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    dataset = Planetoid(dir, data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    return data

def parse_cora():
    path = dir + '/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_Y, np.unique(data_edges, axis=0).transpose()


def load_cora(per_classnum=10, seed=0, thred=1.0, lam=-1, model_type=''):
    data = get_cora_casestudy(seed)
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

    new_embed = np.load(dir + '/cora_process/new_sample.npy')
    text_embed = np.load(dir + '/cora_process/cora_text_embed.npy')
    N = data.num_nodes
    M = int(new_embed.shape[0])
    C = data.y.max() + 1
    K = int(new_embed.shape[0] / C)
    new_y = []
    for i in range(C):
        new_y = new_y + [i]*K
    data.y = torch.cat((data.y, torch.tensor(new_y)))
    data.train_id = data.train_id + list(range(N, N+M))
    data.num_nodes = N + M
    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

   
    if model_type == 'Edge':
        update_edge = data.edge_index
        E = update_edge.size(1)
        sparse_matrix = torch.sparse.FloatTensor(data.edge_index,
            torch.ones(data.edge_index.size(1)),
            torch.Size([N, N]))
        adj = sparse_matrix.to_dense()
        missing_edge = torch.nonzero(adj != 1)
        missing_edge_idx = np.arange(0, missing_edge.size(0))
        np.random.shuffle(missing_edge_idx)
        missing_edge_index = missing_edge[missing_edge_idx[0:E]].transpose(0, 1)
        data.edge_y = torch.cat([torch.ones(E), torch.zeros(E)])
        data.train_edge = torch.cat([update_edge, missing_edge_index], dim=1)
        
        # Filtering predicting pairs
        sim = torch.mm(torch.from_numpy(new_embed), torch.from_numpy(text_embed).t())
        new_edge = torch.nonzero(sim > thred)
        new_edge[:,0] = new_edge[:,0] + N
        data.predict_edge_index = new_edge.transpose(0,1)
        print('predict size:', data.predict_edge_index.size())
    elif model_type == 'Node':
        if lam == -1:
            pass
        else:
            data.edge_index = torch.load('./dataset/cora_process/adj/lam_' + str(lam) + '_thred_' + str(thred) + '.pt')
    else:
        print('error')
        exit()
  
    return data, np.concatenate((text_embed, new_embed))
    
    