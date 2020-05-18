import os
num_thread = 2
os.environ["OMP_NUM_THREADS"] = str(num_thread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(num_thread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_thread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(num_thread) # export NUMEXPR_NUM_THREADS=1

import dgl
import pandas as pd 
import numpy as np
import torch
#from model_folder.TGAT import *
import random
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import argparse
import time
import datetime
import math
import sys
import pickle
import copy
import shutil
import importlib
import scipy.sparse as spp
from torch.nn.utils.rnn import pack_sequence
import bisect

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i-1
    return -1

seed_num = 0
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')

parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia_small')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='the name of this setting')
parser.add_argument('--tbatch_num', type=int, default=3, help='tbatch_num')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--neg_sampling_ratio', type=int, default=1, help='The ratio of negative sampling')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle the tbatch')
parser.add_argument('--expand_factor', type=int, default=20, help='sampling neighborhood size')
parser.add_argument('--model_file', type=str, default="TGAT_nf_v2", help='the model file')
parser.add_argument('--encoding', type=str, default="temporal", help='temporal encoding method')
parser.add_argument('--rnn_layer', type=int, default=1, help='the number of RNN layers')
parser.add_argument('--snapshot_num', type=int, default=64, help='the maximum number of snapshot the model would consider (most recent)')
parser.add_argument('--rnn_bidirection', type=bool, default=False, help='whether use bidirection RNN to aggregate neighborhood (if applicable)')
parser.add_argument('--local_rnn_layer', type=int, default=1, help='the number of layer for RNN to aggregate neighborhood (if applicable)')
parser.add_argument('--local_rnn_bidirection', type=bool, default=False, help='whether use bidirection RNN to aggregate neighborhood (if applicable)')
args = parser.parse_args()


model_file = args.model_file
model_module = importlib.import_module("model_folder."+model_file)
class Logger():
    def __init__(self, lognames):
        self.terminal = sys.stdout
        self.logs = []
        for log_name in lognames:
            self.logs.append(open(log_name, 'w'))
    def write(self, message):
        self.terminal.write(message)
        for log in self.logs:
            log.write(message)
            log.flush()
    def flush(self):
        pass

dataset_name = args.data
setting_name = "TGAT-L"+args.name
device = args.device#"cuda:0"#"cpu"
log_dir = str(dataset_name+"_"+setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
os.mkdir(log_dir)
sys.stdout = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
sys.stderr = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
snapshot_dir = os.path.join(log_dir, "snapshot")
if not os.path.isdir(snapshot_dir):
    os.makedirs(snapshot_dir)
    
print("Process Id:", os.getpid())
print(os.path.join(log_dir, sys.argv[0]))
print(args)

shutil.copyfile(__file__, os.path.join(log_dir, "train.py"))
shutil.copyfile(os.path.join("model_folder", model_file+".py"), os.path.join(log_dir, model_file+".py"))


linkage_df = pd.read_csv("./processed/processed_linkage_{}.csv".format(dataset_name), index_col=0)
#src, dst, t, label
edge_feature = np.load('./processed/processed_edge_feat_{}.npy'.format(dataset_name))
node_feature = np.load('./processed/processed_node_feat_{}.npy'.format(dataset_name))
num_node = max(linkage_df.u.max(), linkage_df.i.max())+1
num_edge = linkage_df.shape[0]

print("Node Num:", num_node, "Edge Num:", num_edge, "edge_feature_dim", edge_feature.shape[1])

##Normalize Features
linkage_df.ts = (linkage_df.ts - linkage_df.ts.mean()) / linkage_df.ts.std()
edge_feature = (edge_feature - edge_feature.mean(axis=0)) / (edge_feature.std(axis=0)+1e-17)
node_feature = (node_feature - node_feature.mean(axis=0)) / (node_feature.std(axis=0)+1e-17)
#linkage_df["feat"] = edge_feature.tolist()
node_feature = torch.Tensor(node_feature)#.to(device)
edge_feature = torch.Tensor(edge_feature)#.to(device)
timestamp_feature = torch.Tensor(linkage_df.ts).unsqueeze(-1)#.to(device)

### Train Val Test 0.7 - 0.15 - 0.15
entire_start_timestamp = float(linkage_df.ts.min())
val_start_timetamp, test_start_timetamp = list(np.quantile(linkage_df.ts, [0.70, 0.85]))
entire_end_timestamp = float(linkage_df.ts.max())

train_edge = linkage_df[linkage_df.ts<val_start_timetamp]
val_edge =  linkage_df[(linkage_df.ts>val_start_timetamp) & (linkage_df.ts<test_start_timetamp)]
test_edge = linkage_df[linkage_df.ts>test_start_timetamp]

###mask
#all_i_node = np.unique(np.array(linkage_df.i))
###For reddit/wiki dataset !!!
i_min_index = linkage_df.i[0] ###negative sampling

all_val_u_node = set(list(val_edge.u) + list(test_edge.u))
all_val_i_node = set(list(val_edge.i) + list(test_edge.i))
masked_u_node = random.sample(all_val_u_node, int(len(all_val_u_node)*0.1))
masked_i_node = random.sample(all_val_i_node, int(len(all_val_i_node)*0.1))
train_edge = train_edge[(~train_edge['u'].isin(masked_u_node))&(~train_edge['i'].isin(masked_i_node))]
all_edge = pd.concat([train_edge, val_edge, test_edge])

tbatch_num = args.tbatch_num#500.0
tbatch_timespan = float((train_edge.ts.max()-train_edge.ts.min())/float(tbatch_num))

train_snapshot_num = tbatch_num ##move the rest to val_edge
val_snapshot_num = int((val_edge.ts.max()-val_edge.ts.min())/tbatch_timespan)+1
test_snapshot_num = int((test_edge.ts.max()-test_edge.ts.min())/tbatch_timespan)+1

###With cache
snapshot_dic = {}
src_node_existing_index = {}
dst_node_existing_index = {} 


def build_snapshot_dic(time_index, state):
    global snapshot_dic, src_node_existing_index, dst_node_existing_index
    tmp_dic = {}
    now_time = entire_start_timestamp + (time_index+1) * tbatch_timespan
    now_all_edge = all_edge[(all_edge.ts<now_time) & (all_edge.ts >= (now_time-tbatch_timespan))].copy()
    if now_all_edge.shape[0] == 0:
        snapshot_dic[time_index] = None
        return
    now_src_node = list(now_all_edge.u.unique())
    for n in now_src_node:
        if n not in src_node_existing_index:
            src_node_existing_index[n] = []
        src_node_existing_index[n].append(time_index)
    now_dst_node = list(now_all_edge.i.unique())
    for n in now_dst_node:
        if n not in dst_node_existing_index:
            dst_node_existing_index[n] = []
        dst_node_existing_index[n].append(time_index)
    now_all_node = now_src_node+now_dst_node
    now_all_node.sort()
    node_index_global2local = {}
    node_index_local2global = {}
    for i in range(len(now_all_node)):
        node_index_global2local[now_all_node[i]] = i
        node_index_local2global[i] = now_all_node[i]
    now_all_local_u = np.array(now_all_edge['u'].map(node_index_global2local))
    now_all_local_i = np.array(now_all_edge['i'].map(node_index_global2local))
    
    latest_edge = now_all_edge.groupby(["u"]).tail(1)
    latest_u_local = np.array(latest_edge['u'].map(node_index_global2local))
    latest_i_local = np.array(latest_edge["i"].map(node_index_global2local))
    
    #add bidirection + add self-loop
    adj = spp.coo_matrix((np.ones(int(now_all_local_u.shape[0]+now_all_local_i.shape[0]+len(now_all_node))), (np.concatenate([now_all_local_u, now_all_local_i, list(range(len(now_all_node)))]), np.concatenate([now_all_local_i, now_all_local_u, list(range(len(now_all_node)))]))))
    now_graph = dgl.DGLGraph(adj)
    now_graph.readonly()
    now_graph.ndata["node_raw_feat"] = node_feature[now_all_node]
    now_all_edge_feat = edge_feature[np.array(now_all_edge.index)]
    now_graph.edata["edge_raw_feat"] = torch.cat([now_all_edge_feat, now_all_edge_feat, torch.zeros(len(now_all_node), edge_feature.shape[-1])], dim=0) ###bidirected
    now_all_edge_t = timestamp_feature[np.array(now_all_edge.index)]
    now_graph.edata["t"] = torch.cat([now_all_edge_t, now_all_edge_t, torch.zeros(len(now_all_node), timestamp_feature.shape[-1])], dim=0)

    tmp_dic["induct_edge_lis_bool"] = None
    if state != "Train":
        induct_u_node =  latest_edge['u'].isin(masked_u_node).to_numpy()
        induct_i_node = latest_edge['i'].isin(masked_i_node).to_numpy()
        induct_edge_lis = induct_u_node | induct_i_node
        tmp_dic["induct_edge_lis_bool"] = induct_edge_lis
        
    tmp_dic["graph"] = now_graph
    #tmp_dic["now_all_node"] = now_all_node
    #tmp_dic["now_all_edge"] = np.array(now_all_edge.index)
    #tmp_dic["now_dst_node_local"] = set(now_all_local_i.tolist())
    tmp_dic["t_now"] = now_time
    tmp_dic["node_index_global2local"] = node_index_global2local
    #tmp_dic["now_dst_local"] = set(now_all_local_i.tolist())
    #tmp_dic["new_src_local"] = latest_u_local
    #tmp_dic["new_dst_local"] = latest_i_local
    tmp_dic["now_dst"] = set(now_dst_node)
    tmp_dic["new_src"] = np.array(latest_edge['u'])
    tmp_dic["new_dst"] = np.array(latest_edge['i'])
    tmp_dic["new_src_node_label"] = np.array(latest_edge.label)
    snapshot_dic[time_index] = tmp_dic


for time_index in range(train_snapshot_num):
    build_snapshot_dic(time_index, state="Train")
for time_index in range(train_snapshot_num, train_snapshot_num+val_snapshot_num):
    build_snapshot_dic(time_index, state="Val")
for time_index in range(train_snapshot_num+val_snapshot_num, train_snapshot_num+val_snapshot_num+test_snapshot_num):
    build_snapshot_dic(time_index, state="Test")

#print(src_node_existing_index, src_node_time_index_cnt)
#exit()
neg_sampling_ratio = args.neg_sampling_ratio

gnn_model = model_module.TGAT_nf(num_layers=args.n_layer, n_head=args.n_head, node_dim=node_feature.shape[-1], d_k=node_feature.shape[-1]//args.n_head, d_v=node_feature.shape[-1]//args.n_head, d_T=node_feature.shape[-1], args=args, edge_dim=node_feature.shape[-1], device=device, dropout=args.drop_out)
rnn_model = torch.nn.LSTM(input_size=node_feature.shape[-1], hidden_size=node_feature.shape[-1], num_layers=args.rnn_layer, bidirectional=args.rnn_bidirection, batch_first=True)
if args.rnn_bidirection:
    link_classifier = model_module.MergeLayer(node_feature.shape[-1]*2, node_feature.shape[-1]*2, node_feature.shape[-1]*2, 1)
else:
    link_classifier = model_module.MergeLayer(node_feature.shape[-1], node_feature.shape[-1], node_feature.shape[-1], 1)
optimizer = torch.optim.Adam(list(gnn_model.parameters())+list(link_classifier.parameters()), lr=args.lr)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
gnn_model.to(device)
rnn_model.to(device)
link_classifier.to(device)

optimizer = torch.optim.Adam(list(gnn_model.parameters())+list(rnn_model.parameters())+list(link_classifier.parameters()), lr=args.lr)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

epoch_num = args.n_epoch
best_val_ap = 0
best_val_ap_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0

def run_model(state):
    global best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch
    all_acc, all_ap, all_f1, all_auc, all_m_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    transduct_acc, transduct_ap = AverageMeter(), AverageMeter()
    induct_acc, induct_ap = AverageMeter(), AverageMeter()
    if state == "Train":
        is_train = True
        if "small" not in args.data: ##not in writing code mode
            start_index = 5 ##do not train with very first graph
        else:
            start_index = 0
        end_index = train_snapshot_num
    if state == "Val":
        is_train = False
        start_index = train_snapshot_num
        end_index = train_snapshot_num + test_snapshot_num
    if state == "Test":
        is_train = False
        start_index = train_snapshot_num + test_snapshot_num
        end_index = train_snapshot_num + test_snapshot_num + val_snapshot_num
    tbatch_index = list(range(start_index, end_index))
    print_freq = len(tbatch_index) // 3 + 1
    if args.shuffle:
        random.shuffle(tbatch_index)
    start_time = time.time()
    if is_train:
        gnn_model.train()
        rnn_model.train()
        link_classifier.train()
    else:
        gnn_model.eval()
        rnn_model.eval()
        link_classifier.eval()
    with torch.set_grad_enabled(is_train):
        for index, time_index in enumerate(tbatch_index):
            ###Find all its previous snapshots
            with torch.no_grad():
                tmp_dic = snapshot_dic[time_index]
                if tmp_dic is None:
                    continue
                new_src = tmp_dic["new_src"] ## the nodes we use as train/eval/test sample - global index
                new_dst = tmp_dic["new_dst"]
                now_dst = tmp_dic["now_dst"]
                induct_edge_lis_bool = tmp_dic["induct_edge_lis_bool"]
                neg_dst_node = np.array(np.array([random.sample(list(range(i_min_index, new_dst[i]))+list(now_dst-set([new_dst[i]])), neg_sampling_ratio) for i in range(len(new_dst))]))
                sample_dst_node = np.concatenate([np.expand_dims(new_dst, axis=-1), neg_dst_node], axis=-1).ravel()
                label = torch.Tensor(([1]+[0]*neg_sampling_ratio)*new_src.shape[0]).to(device)

            #snapshot_feature_lis = []
            prob = []
            optimizer.zero_grad()
            for src_node_i in range(len(new_src)):
                src_now_feature = []
                src_num_of_previous_snapeshots = find_le(src_node_existing_index[new_src[src_node_i]], time_index)+1#src_node_time_index_cnt[new_src[src_node_i]][time_index]
                src_previous_time_index_list = src_node_existing_index[new_src[src_node_i]][max(0, src_num_of_previous_snapeshots-args.snapshot_num):src_num_of_previous_snapeshots]
                #snapshot_legnth_lis.append(len(previous_time_index_list))
                for previous_time_index in src_previous_time_index_list:
                    for nf in dgl.contrib.sampling.NeighborSampler(g=snapshot_dic[previous_time_index]["graph"], batch_size=1, expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=torch.Tensor([snapshot_dic[previous_time_index]["node_index_global2local"][new_src[src_node_i]]]).type(torch.int64), num_workers=1):
                        nf.copy_from_parent(ctx=torch.device(device))
                        src_now_feature.append(gnn_model(nf, snapshot_dic[previous_time_index]["t_now"]))
                src_now_feature, (_,_) = rnn_model(torch.stack(src_now_feature, 1))
                src_now_feature = src_now_feature[0,-1,:]
                
                dst_feature_dic = {}
                for dst_node_j in range(args.neg_sampling_ratio+1):
                    dst_now_feature = []
                    dst_now_index = sample_dst_node[src_node_i*(args.neg_sampling_ratio+1)+dst_node_j]
                    ##Not cached
                    if dst_now_index not in dst_feature_dic:
                        dst_num_of_previous_snapeshots = find_le(dst_node_existing_index[dst_now_index], time_index)+1
                        ##No history
                        if dst_num_of_previous_snapeshots == 0:
                            dst_feature_dic[dst_now_index] = None
                            prob.append(torch.Tensor([0]).to(device))
                            continue
                        dst_previous_time_index_list = dst_node_existing_index[dst_now_index][max(0, dst_num_of_previous_snapeshots-args.snapshot_num):dst_num_of_previous_snapeshots]
                        #print(dst_num_of_previous_snapeshots, max(0, dst_num_of_previous_snapeshots-args.snapshot_num), dst_num_of_previous_snapeshots, len(dst_previous_time_index_list), len(dst_node_existing_index[dst_now_index]))
                        for previous_time_index in dst_previous_time_index_list:
                            for nf in dgl.contrib.sampling.NeighborSampler(g=snapshot_dic[previous_time_index]["graph"], batch_size=1, expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=torch.Tensor([snapshot_dic[previous_time_index]["node_index_global2local"][dst_now_index]]).type(torch.int64), num_workers=1):
                                nf.copy_from_parent(ctx=torch.device(device))
                                dst_now_feature.append(gnn_model(nf, snapshot_dic[previous_time_index]["t_now"]))
                        dst_now_feature, (_,_) = rnn_model(torch.stack(dst_now_feature, 1))
                        dst_now_feature = dst_now_feature[0,-1,:]
                        dst_feature_dic[dst_now_index] = dst_now_feature
                    ###No history
                    if dst_feature_dic[dst_now_index] is None:
                        prob.append(torch.Tensor([0]).to(device))
                        continue
                    prob.append(link_classifier(src_now_feature, dst_now_feature))
            prob = torch.cat(prob).squeeze()
            loss = criterion(prob, label)
            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                label = label.cpu().detach().numpy()
                pred_score = prob.cpu().detach().numpy()
                pred_label = pred_score > 0.5
                all_ap.update(average_precision_score(label, pred_score), label.shape[0])
                all_acc.update((pred_label==label).mean(), label.shape[0])
                all_m_loss.update(loss.item(), label.shape[0])
                all_auc.update(roc_auc_score(label, pred_score), label.shape[0])
                all_f1.update(f1_score(label, pred_label), label.shape[0])
                if state != "Train":
                    induct_edge_lis_bool = np.repeat(induct_edge_lis_bool, args.neg_sampling_ratio+1)
                    trasduct_score = pred_score[~induct_edge_lis_bool]
                    if trasduct_score.shape[0] != 0:
                        transduct_acc.update((pred_label[~induct_edge_lis_bool]==label[~induct_edge_lis_bool]).mean(), trasduct_score.shape[0])
                        transduct_ap.update(average_precision_score(label[~induct_edge_lis_bool], trasduct_score), trasduct_score.shape[0])
                    induct_score = pred_score[induct_edge_lis_bool]
                    if induct_score.shape[0] != 0:
                        induct_acc.update((pred_label[induct_edge_lis_bool]==label[induct_edge_lis_bool]).mean(), induct_score.shape[0])
                        induct_ap.update(average_precision_score(label[induct_edge_lis_bool], induct_score), induct_score.shape[0])
            if is_train & ((index+1) % print_freq == 0):
                print('Train Epoch: [{0}][{1}/{2}][{all_ap.count}({3})], AP {all_ap.val:.4f}({all_ap.avg:.4f}), Acc {all_acc.val:.4f}({all_acc.avg:.4f}), Loss {all_m_loss.val:.2e}({all_m_loss.avg:.2e}), Auc {all_auc.val:.4f}({all_auc.avg:.4f}), F1 {all_f1.val:.4f}({all_f1.avg:.4f})'.format(epoch, index+1, len(tbatch_index), new_src.shape[0], all_ap=all_ap, all_acc=all_acc, all_m_loss=all_m_loss, all_auc=all_auc, all_f1=all_f1))

    print_text = state +' Epoch: [{0}][{1}/{2}][{all_ap.count}], AP {all_ap.val:.4f}({all_ap.avg:.4f}), Acc {all_acc.val:.4f}({all_acc.avg:.4f}), Loss {all_m_loss.val:.2e}({all_m_loss.avg:.2e}), Auc {all_auc.val:.4f}({all_auc.avg:.4f}), F1 {all_f1.val:.4f}({all_f1.avg:.4f})'.format(epoch, len(tbatch_index), len(tbatch_index), all_ap=all_ap, all_acc=all_acc, all_m_loss=all_m_loss, all_auc=all_auc, all_f1=all_f1)
    
    if state == "Val":
        if best_val_acc < all_acc.avg:
            best_val_acc = all_acc.avg
            best_val_acc_epoch = epoch
        if best_val_ap < all_ap.avg:
            best_val_ap = all_ap.avg
            best_val_ap_epoch = epoch
        print_text =  "**** " + print_text + ", Tran_Acc {transduct_acc.val:.4f}({transduct_acc.avg:.4f}), Tran_AP {transduct_ap.val:.4f}({transduct_ap.avg:.4f}), Indu_Acc {induct_acc.val:.4f}({induct_acc.avg:.4f}), Indu_AP {induct_ap.val:.4f}({induct_ap.avg:.4f})\n".format(transduct_acc=transduct_acc, transduct_ap=transduct_ap, induct_acc=induct_acc, induct_ap=induct_ap) + "Best Val: Acc {1:.4f} (Epoch {2}), AP {3:.4f} (Epoch {4})".format(epoch, best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch)
    if state == "Test":
        print_text = "!!!! " + print_text + ", Tran_Acc {transduct_acc.val:.4f}({transduct_acc.avg:.4f}), Tran_AP {transduct_ap.val:.4f}({transduct_ap.avg:.4f}), Indu_Acc {induct_acc.val:.4f}({induct_acc.avg:.4f}), Indu_AP {induct_ap.val:.4f}({induct_ap.avg:.4f})".format(transduct_acc=transduct_acc, transduct_ap=transduct_ap, induct_acc=induct_acc, induct_ap=induct_ap)
    print(print_text)
    print("Time:", time.time()-start_time)

            
            
for epoch in range(1, args.n_epoch+1):
    #Train
    run_model(state = "Train")
    if epoch % args.val_interval == 0:
        #print("Eval Epoch %d"%(epoch))
        run_model(state = "Val")
    if epoch % args.test_interval == 0:
        #print("Test Epoch %d"%(epoch))
        run_model(state ="Test")
    if epoch % args.snapshot_interval == 0:
        print("Epoch %d Save Model"%(epoch))
        file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_gnn.model")
        file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_rnn.model")
        file_path3 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_link_cls.model")
        #if torch.cuda.is_available():
        torch.save(gnn_model.state_dict(), file_path1)
        torch.save(rnn_model.state_dict(), file_path1)
        torch.save(link_classifier.state_dict(), file_path3)
        # else:
        #     torch.save(gnn_model.state_dict(), file_path1)
        #     torch.save(rnn_model.state_dict(), file_path1)
        #     torch.save(link_classifier.state_dict(), file_path3)