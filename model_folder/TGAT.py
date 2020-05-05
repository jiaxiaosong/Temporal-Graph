import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn) # n_head, 
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0) #1
        seq_len = ts.size(1) #0

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)

class MergeLayer(torch.nn.Module):

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TGATLayer(nn.Module):
    def __init__(self, n_head, node_dim, d_k, d_v, d_T, edge_dim=None, dropout=0.1, act=nn.LeakyReLU(negative_slope=0.2), device="cpu"):
        super(TGATLayer, self).__init__()

        ### Multi-head Atnn
        self.n_head = n_head
        self.act = act
        self.attn_dropot = dropout
        self.d_k = d_k
        self.d_v = d_v
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.device = device
        
        d_model = node_dim + d_T
        if  self.edge_dim:
            d_model +=  self.edge_dim
        self.d_model = d_model
        ##TGAT's authors simply let d_model//n_head = d_k and make sure d_model%n_head = 0
        #assert(n_head * d_k == d_model)
        #assert(d_k == d_v)
        if n_head * d_k != d_model:
            d_k = d_model//n_head
            d_v = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn_fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.attn_fc.weight)
        self.attn_dropout = nn.Dropout(dropout)

        ##Temporal Encoding
        self.d_T = d_T
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_T))).float()).view(1, -1).to(device)
        self.phase = torch.nn.Parameter(torch.zeros(self.d_T).float()).to(device)
        self.t_now = None

        ###Merger
        self.merger = MergeLayer(self.d_model, node_dim, node_dim, node_dim)

    def forward(self, g, t_now, layer_index):
        self.t_now = t_now
        if layer_index == 0:
            g.ndata['node_h'] = g.ndata['node_raw_feat']

        def message_func(edges):
            #edge_features: time-stamp
            t_encoding = self.t_now - edges.data["t"] #Edge_Num, 1
            t_encoding = t_encoding * self.basis_freq + self.phase #edge_batch, d_t
            t_encoding = torch.cos(t_encoding)

            ##node features edges.src['h']) edge_batch, node_dim
            if self.edge_dim:
                z = torch.cat([edges.src['node_h'], edges.data["edge_raw_feat"], t_encoding], dim=-1) #edge_batch, d_model (d_model = n_head*d_v)
            else:
                z = torch.cat([edges.src['node_h'],  t_encoding], dim=-1) #edge_batch, d_model (d_model = n_head*d_v)
            k = self.w_ks(z) # edge_batch, n_head*d_k
            v = self.w_vs(z) # edge_batch, n_head*d_v
            return {"k":k, "v":v}

        def reduce_func(nodes):
            ## mail_box_features: node_batch, neightbor_num, n_head*d_model
            node_batch, neightbor_num, _ = nodes.mailbox['k'].size()
            ##self-loop term
            if self.edge_dim:
                self_z = torch.cat([nodes.data['node_h'], torch.zeros(node_batch, self.edge_dim, device=self.device), torch.ones(node_batch, self.d_T, device=self.device)*torch.cos(self.phase)], dim=-1).unsqueeze(1)#node_batch, 1, d_model
            else:
                self_z = torch.cat([nodes.data['node_h'], torch.ones(node_batch, self.d_T, device=self.device)*torch.cos(self.phase)], dim=-1).unsqueeze(1) #node_batch, 1, d_model

            #z = torch.cat([self_z, nodes.mailbox['z']], dim=1).view(node_batch, neightbor_num+1, self.n_head, -1) #node_batch, neightbor_num+1, n_head, d_model
            q = self.w_qs(self_z).view(node_batch, 1, self.n_head, -1) #node_batch, 1, n_head, d_model
            k = nodes.mailbox['k'].view(node_batch, neightbor_num, self.n_head, -1)#node_batch, neightbor_num, n_head, d_model
            v = nodes.mailbox['v'].view(node_batch, neightbor_num, self.n_head, -1)#node_batch, neightbor_num, n_head, d_model

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.d_model)  # (n_head*node_batch), 1, d_model
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_model)  # (n_head*node_batch), neightbor_num, d_model
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_model)  # (n_head*node_batch), neightbor_num, d_model

            output, attn = self.attention(q, k, v, mask=None) #(n_head*node_batch), d_model
            output = output.squeeze(1).view(self.n_head, node_batch, -1).permute(1, 0, 2).contiguous().view(node_batch, -1)#node_batch, n_head*d_k

            output = self.attn_dropout(self.attn_fc(output)) ##node_batch, d_model
            output = self.layer_norm(output + self_z.squeeze(1)) ## node_batch, node_dim
            
            ###Merger
            output = self.merger(output, nodes.data['node_h']) ## node_batch, node_dim
            return {'node_h':output}
        g.update_all(message_func, reduce_func)
        

class TGAT(nn.Module):
    def __init__(self, num_layers, n_head, node_dim, d_k, d_v, d_T, edge_dim = None, dropout=0.1, act=nn.LeakyReLU(negative_slope=0.2), device="cpu"):
        super(TGAT, self).__init__()
        self.gnn_layers = torch.nn.ModuleList([TGATLayer(n_head, node_dim, d_k, d_v, d_T, edge_dim, dropout, act, device=device) for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, g, t_now):
        for i in range(self.num_layers):
            self.gnn_layers[i](g, t_now, i)
        output =  g.ndata.pop('node_h')
        return output

#P17 Transductive, we combine the node embedding with the interacting node embedding it is interacting to do prediction by Concatenation


##P8 in the link prediction tasks, we first sample an equal amount of negative node pairs to the positive links and then compute the average precision (AP) and classification accuracy. In the downstream node classification tasks, due to the label imbalance in the datasets, we employ the area under the ROC curve (AUC).

###For node; lr_criterion = torch.nn.BCELoss(); stratified sampling: num of neg == pos; ONLY TRAIN LR!!!!  LR([node embedding h, node features x0])
class LR(torch.nn.Module):
    def __init__(self, node_dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(node_dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_3.weight)
        self.dropout = torch.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).sigmoid()



###For edge self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) -> sigmoid; Loss: sample 1 neg node...





