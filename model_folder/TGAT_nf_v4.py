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
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0) #1
        seq_len = ts.size(1) #0
        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class TGATLayer_v4(nn.Module):
    def __init__(self, n_head, node_dim, d_k, d_v, args, edge_dim=None, dropout=0.1, act=torch.nn.functional.gelu, device="cpu"):
        super(TGATLayer_v4, self).__init__()
        ### Multi-head Atnn
        self.n_head = n_head
        self.act = act
        #self.attn_dropot = dropout
        self.d_k = d_k
        self.d_v = d_v
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.device = device
        self.encoding = args.encoding
        d_model = node_dim
        if  self.edge_dim:
            self.edge_fc = nn.Linear(self.edge_dim, self.node_dim)
            nn.init.xavier_normal_(self.edge_fc.weight)
        self.d_model = d_model
        #if n_head * d_k != d_model:
            #d_k = d_model//n_head
            #d_v = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_qs_src = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks_src = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs_src = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_qs_src.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks_src.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs_src.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.attn_fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.attn_fc.weight)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model*4, dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)


        ##Temporal Encoding
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float()).view(1, -1).to(device)
        self.phase = torch.nn.Parameter(torch.zeros(self.d_model).float()).to(device)
        self.t_now = None

        ###Aggregate Neighbor
        self.fea2node = nn.Linear(d_model*2, self.node_dim)
        nn.init.xavier_normal_(self.fea2node.weight)
        self.layer_norm = nn.LayerNorm(self.node_dim)

    def message_func(self, edges):
        if self.encoding == "temporal":
            #edge_features: time-stamp
            t_encoding = self.t_now - edges.data["t"]#Edge_Num, 1
            t_encoding = t_encoding * self.basis_freq + self.phase #edge_batch, d_t
            t_encoding = torch.cos(t_encoding)
        if self.encoding == 'none':
            t_encoding = torch.zeros(edges.src['node_h'].shape[0], self.d_model).to(self.device)

        ##node features edges.src['h']) edge_batch, node_dim
        if self.edge_dim:
            z = edges.src['node_h'] + self.act(self.edge_fc(edges.data["edge_raw_feat"])) + t_encoding #edge_batch, d_model (d_model = n_head*d_v)
        else:
            z = edges.src['node_h'] + t_encoding#edge_batch, d_model (d_model = n_head*d_v)
        q = self.w_qs(z) # edge_batch, n_head*d_k
        k = self.w_ks(z) # edge_batch, n_head*d_k
        v = self.w_vs(z) # edge_batch, n_head*d_v
        return {"q":q, "k":k, "v":v, "z":z, "lst_node_h":edges.src['node_h']}

    def reduce_func(self, nodes):
        node_batch, neightbor_num, _ = nodes.mailbox['q'].size()

        #### node itself should have different projection
        self_q = self.w_qs_src(nodes.mailbox['z'][:,-1,:]).unsqueeze(1) 
        self_k = self.w_ks_src(nodes.mailbox['z'][:,-1,:]).unsqueeze(1)
        self_v = self.w_vs_src(nodes.mailbox['z'][:,-1,:]).unsqueeze(1)
        q = torch.cat([nodes.mailbox['q'][:,:-1,:], self_q], dim=1) ##node_batch, neightbor_num, n_head*d_k
        k = torch.cat([nodes.mailbox['k'][:,:-1,:], self_q], dim=1) ##node_batch, neightbor_num, n_head*d_k
        v = torch.cat([nodes.mailbox['v'][:,:-1,:], self_q], dim=1) ##node_batch, neightbor_num, n_head*d_v
        
        q = q.view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_k) # (n_head*node_batch), neightbor_num, d_k
        k = k.view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_k) # (n_head*node_batch), neightbor_num, d_k
        v = v.view(node_batch, neightbor_num, self.n_head, -1).permute(2, 0, 1, 3).contiguous().view(-1, neightbor_num, self.d_v) # (n_head*node_batch), neightbor_num, d_v

        output, attn = self.attention(q, k, v, mask=None) #(n_head*node_batch), neighbor_num, d_v
        output = output.view(self.n_head, node_batch, neightbor_num, -1).permute(1, 2, 0, 3).contiguous().view(node_batch, neightbor_num, self.n_head*self.d_v)  #node_batch, neighbor_num, d_v * n_head
        output = self.attn_dropout(self.attn_fc(output)) #node_batch, neighbor_num, d_model
        output = self.attn_layer_norm(output + nodes.mailbox['z']) 
        output = self.pos_ffn(output) #node_batch, neighbor_num, d_model

        neighborhood = output[:,:-1,:].mean(dim=1) #node_batch, d_model
        output = self.layer_norm(self.act(self.fea2node(torch.cat([output[:,-1,:],neighborhood], dim=-1))) + nodes.mailbox['lst_node_h'][:,-1,:])
        return {"node_h":output}

class TGAT_nf(nn.Module):
    def __init__(self, num_layers, n_head, node_dim, d_k, d_v, d_T, args, edge_dim = None, dropout=0.1, act=torch.nn.functional.gelu, device="cpu"):
        super(TGAT_nf, self).__init__()
        self.gnn_layers =  torch.nn.ModuleList([TGATLayer_v4(n_head, node_dim, d_k, d_v, args, edge_dim, dropout, act, device=device) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.device = device
    def forward(self, nf, t_now):
        nf.layers[0].data['node_h'] = nf.layers[0].data['node_raw_feat']
        for i in range(self.num_layers):
            self.gnn_layers[i].t_now = t_now
            nf.block_compute(i, message_func=self.gnn_layers[i].message_func, reduce_func=self.gnn_layers[i].reduce_func)
        
        return nf.layers[-1].data.pop('node_h')


class LR(torch.nn.Module):
    def __init__(self, node_dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(node_dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_3.weight)
        self.dropout = torch.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x)


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