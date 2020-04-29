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


class TGATLayer(nn.Module):
    def __init__(self, n_head, f_in, d_k, d_v, d_T, dropout=0.1, act=nn.LeakyReLU(negative_slope=0.2)):
        super(TGATLayer, self).__init__()

        ### Multi-head Atnn
        self.n_head = n_head
        self.act = act
        self.attn_dropot = dropout
        self.d_k = d_k
        self.d_v = d_v
        
        d_model = f_in + d_T
        ##TGAT's authors simply let d_model//n_head = d_k and make sure d_model%n_head = 0
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn_fc = nn.Linear(n_head * d_v, d_model - d_T)
        nn.init.xavier_normal_(self.fc.weight)
        self.attn_dropout = nn.Dropout(dropout)

        ##Temporal Encoding
        self.d_T = d_T
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_T))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.d_T).float())

        self.t_now = None

    def forward(self, g, t_now):
        self.t_now = t_now
        def message_func(edges):
            #edge_features: time-stamp !!!! how to access the edge features, what's its dimensions? what's edges.src['h']'s dimension?
            t_encoding = self.t - edges.data['w'].unsqueeze(-1) #Edge_Num, 1
            t_encoding = t_encoding * self.basis_freq.view(1, -1) + self.phas #Edge_Num, d_t
            t_encoding = torch.cos(t_encoding)

            ##node features edges.src['h']) Edge_Num, f_in
            z = torch.cat([edges.src['h'], t_encoding], dim=-1) #Edge_Num, d_model (d_model = n_head*d_k)
            edges_num, d_model = z.size()
            z = z.view(edges_num, self.n_head, -1) ##Edge_Num, n_head, d_k as in the author's code, they simply divide the concatenated feature by head, which is a little unreasonable

            q = self.w_qs(z) # Edge_Num, n_head, d_k
            k = self.w_ks(z) # Edge_Num, n_head, d_k
            v = self.w_vs(z) # Edge_Num, n_head, d_k
            return {'q':q, "k":k, "v":v, 'z':z}

        def reduce_func(nodes):
            ## mail_box_features: n_head, d_k, neightbor_num
            q = nodes.mailbox['q']
            .permute(0, 2, 1).contiguous() ##n_head, neightbor_num, d_k
            k = nodes.mailbox['k'].permute(0, 2, 1).contiguous() ##n_head, neightbor_num, d_k
            v = nodes.mailbox['v'].permute(0, 2, 1).contiguous() ##n_head, neightbor_num, d_k
            
            ##self-loop term  !!!! how to access the node features, what's its dimensions? what's nodes.mailbox['q']'s dimension? 
            #another implementation: add self-loop at each time-step with time-stamp at t_noow
            self_q = nodes.data['h']

            output, attn = self.attention(q, k, v, mask=None) ##n_head, neightbor_num+1, d_k
            output = torch.sum(outputm, dim=1).view(-1) ##n_head,  d_k 
            output = self.dropout(self.attn_fc(output)) ### d_model - d_T = f_in
            
            output = self.layer_norm(output + nodes.data['h']) ## f_in







