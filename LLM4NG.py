import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from load_cora import load_cora
from load_pubmed import load_pubmed
from load_arxiv import load_arxiv
import random
import numpy as np
from utils import set_params

class GCN_Encoder(torch.nn.Module):                           
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim) 
        self.conv2 = GCNConv(hidden_dim,  out_channels) 

    def forward(self, x, edge_index, drop):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=drop, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x ,dim=1)
        return F.log_softmax(x, dim=1)
    
def node_classification():
    args = set_params()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    result = []
    print(args)
    if args.model_type != 'Node':
        print("error")
        exit()
    for s in range(5):
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s) 
        random.seed(s)

        if args.dataset == 'cora':
            data, text_embed = load_cora(per_classnum=args.ratio, seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
        elif args.dataset == 'pubmed':
            data, text_embed = load_pubmed(per_classnum=args.ratio, seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
        elif args.dataset == 'arxiv':
            data, text_embed = load_arxiv(per_classnum=args.ratio, seed=s, thred=args.thred, lam=args.lam, model_type=args.model_type)
        else:
            print("error")
            exit()
        
        # parameters
        hidden_dim = args.hidden
        out_channels = int(data.y.max() + 1)
        dropout = args.dropout
        epochs = 500

        data = data.to(device)
        text_embed = torch.from_numpy(text_embed).to(device)
        input_feature = text_embed

        # model
        num_features = input_feature.size(1)
        Node = GCN_Encoder(num_features, hidden_dim, out_channels).to(device)
        optimizer_n = torch.optim.Adam(Node.parameters(), lr=args.lr, weight_decay=args.l2_coef)


        max_val = 0
        best_test = 0
        
        for epoch in range(1, epochs + 1):
            Node.train()
            optimizer_n.zero_grad()
            out_node = Node(input_feature, data.edge_index, dropout)
            loss_node = F.nll_loss(out_node[data.train_mask], data.y[data.train_mask])
            loss_node.backward()
            optimizer_n.step()

            Node.eval()
            logits = out_node.argmax(dim=1)
            train_acc = logits[data.train_mask].eq(data.y[data.train_mask]).sum() / data.train_mask.sum()
            val_acc = logits[data.val_mask].eq(data.y[data.val_mask]).sum() / data.val_mask.sum()
            test_acc = logits[data.test_mask].eq(data.y[data.test_mask]).sum() / data.test_mask.sum()
            if val_acc > max_val:
                max_val = val_acc
                best_test = test_acc
            if epoch % 5 == 0 :
                pass
                # print(loss_node, args.lam * loss_edge, 0.001 * predict_edge_norm)
                # print('loss:', loss)
                # print('Train_acc: {:.4f},  Val_acc: {:.4f}, Test_acc: {:.4f}'.format(train_acc, val_acc, test_acc))
        # print('Best epoch: {:d}, Best test acc: {:.4f}'.format(max_epoch, best_test))
        result.append(best_test.cpu().detach())

    result = np.array(result)
    print(result)
    print('Ratio:{:d} Results:{:.2f}({:.2f})'.format(args.ratio, result.mean()*100, result.std()*100))

  
if __name__ == '__main__':
    node_classification()