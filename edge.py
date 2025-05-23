import torch
import torch.nn.functional as F
import torch.nn as nn
from load_cora import load_cora
from load_pubmed import load_pubmed
import random
from utils import set_params
import warnings

warnings.filterwarnings("ignore")
from torch.utils.data import Dataset


class Edge_Predictor(torch.nn.Module):                           
    def __init__(self, input_dim, hidden_dim, predict_edge_index):
        super(Edge_Predictor, self).__init__()
                                
        self.MLP = nn.Sequential(
                                nn.Linear(input_dim*2, hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim, 1),
                                 nn.Sigmoid(),
                                )
        self.predict_edge_index = predict_edge_index

    def forward(self, x, train_edge_idx):
        self.x = x
        train_x = self.x[train_edge_idx[0]]
        train_y = self.x[train_edge_idx[1]]
        train_edge = self.MLP(torch.cat([train_x, train_y], dim=1))
        return train_edge

    def predict_bce(self, lam):
        predict_x = self.x[self.predict_edge_index[0]]
        predict_y = self.x[self.predict_edge_index[1]]
        predict_new_edge_kc2n = self.MLP(torch.cat([predict_x, predict_y], dim=1)).squeeze(1)
        predict_new_edge_n2kc = self.MLP(torch.cat([predict_y, predict_x], dim=1)).squeeze(1)
        predict_new_edge = predict_new_edge_kc2n + predict_new_edge_n2kc
        values, indices = torch.topk(predict_new_edge, k=lam)
        predict_edge = self.predict_edge_index[:,indices]
        return predict_edge.detach()


def edge_predictor():
    args = set_params()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.model_type != 'Edge':
        print("error")
        exit()
    s = 1
    if args.dataset == 'cora':
        data, text_embed = load_cora(per_classnum=args.ratio, seed=s, thred=args.thred, model_type=args.model_type)
    elif args.dataset == 'pubmed':
        data, text_embed = load_pubmed(per_classnum=args.ratio, seed=s, thred=args.thred, model_type=args.model_type)
    else:
        print("error")
        exit()
    
    # parameters
    data = data.to(device)
    text_embed = torch.from_numpy(text_embed).to(device)
    input_feature = text_embed
    hidden_dim = 32

    # model
    num_features = input_feature.size(1)
    Edge = Edge_Predictor(num_features, hidden_dim, data.predict_edge_index).to(device)
    optimizer_e = torch.optim.Adam(Edge.parameters(), lr=0.05, weight_decay=0)

    input_edge_index = data.edge_index
    criterion = nn.BCELoss()
    for epoch in range(1, 300):
        Edge.train()
        optimizer_e.zero_grad()
        out_edge = Edge(text_embed, data.train_edge)
        loss_edge = criterion(out_edge, data.edge_y.unsqueeze(1))
        loss_edge.backward()
        optimizer_e.step()

    edge_acc = torch.where(out_edge.squeeze(1)>0.5,1.,0.).eq(data.edge_y).sum() / out_edge.size(0)
    print('Edge: loss:{:.4f} acc:{:.4f}'.format(loss_edge, edge_acc))

    new_edge = Edge.predict_bce(args.lam)
    input_edge_index = torch.cat([data.edge_index, new_edge],dim=1)
    input_edge_index = torch.cat([input_edge_index, new_edge[[1,0]]],dim=1)
    torch.save(input_edge_index.cpu().detach(), './dataset/' + str(args.dataset) + '_process/adj/lam_' + str(args.lam) + '_thred_' + str(args.thred) + '.pt')

if __name__ == '__main__':
    edge_predictor()