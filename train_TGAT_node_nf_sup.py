import os
num_thread = 6
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
from sklearn.metrics import roc_auc_score
import scipy.sparse as spp


model_file = "TGAT_nf"
model_module = importlib.import_module("model_folder."+model_file)
print("Process Id:", os.getpid())
seed_num = 0
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')

parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia_small')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--device', type=str, default="cuda:0", help='idx for the gpu to use')
parser.add_argument('--name', type=str, default="", help='The name of this setting')
parser.add_argument('--pretrained', type=str, default="None", help='The position of pretrained model')
parser.add_argument('--tbatch_num', type=int, default=3, help='tbatch_num')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle the tbatch')
##wiki pos0.14% reddit:0.05%
parser.add_argument('--pos_weight', type=int, default=1, help='weight for positive samples')
parser.add_argument('--expand_factor', type=int, default=20, help='sampling neighborhood size')
args = parser.parse_args()


print(args)
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
setting_name = "TGAT-N"+args.name
device = args.device#"cuda:0"#"cpu"
log_dir = str(dataset_name+"_"+setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
os.mkdir(log_dir)
sys.stdout = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
sys.stderr = Logger(["%s.log"%(dataset_name+setting_name), os.path.join(log_dir, "%s.log"%(dataset_name+"_"+setting_name))])
snapshot_dir = os.path.join(log_dir, "snapshot")
if not os.path.isdir(snapshot_dir):
    os.makedirs(snapshot_dir)
print(os.path.join(log_dir, sys.argv[0]))
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
all_val_u_node = set(list(val_edge.u) + list(test_edge.u))
all_val_i_node = set(list(val_edge.i) + list(test_edge.i))
masked_u_node = random.sample(all_val_u_node, int(len(all_val_u_node)*0.1))
masked_i_node = random.sample(all_val_i_node, int(len(all_val_i_node)*0.1))
train_edge = train_edge[(~train_edge['u'].isin(masked_u_node))&(~train_edge['i'].isin(masked_i_node))]

###Different from the TGAT paper, we use the model in t-batch instead of only one loss for each item-node in one of the three sets. It is the same evaluation method as in the original dataset paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
tbatch_num = args.tbatch_num#500.0
tbatch_timespan = float((train_edge.ts.max()-train_edge.ts.min())/float(tbatch_num))


temporal_graph_dic = {}
def build_temporal_graph_cache(node_feature, edge_list, time_index, start_timestamp, tbatch_timespan, state):
    now_time = start_timestamp + (time_index+1) * tbatch_timespan
    cache_key = state+str(time_index)
    if cache_key not in temporal_graph_dic:
        tmp_dic = {}
        ###"now" prefix means all the node and edge until now_time; 
        ###"new" prefix means the edge appearing during [t_now-timespan, t_now)
        ### "local" means in the index system of local graph; if there is no "local", it means in the index system of global data
        now_all_edge = edge_list[edge_list.ts<=now_time].copy()
        now_src_node = list(now_all_edge.u.unique())
        now_dst_node = list(now_all_edge.i.unique())
        now_all_node = now_src_node + now_dst_node
        now_all_node.sort()
        node_index_global2local = {}
        node_index_local2global = {}
        for i in range(len(now_all_node)):
            node_index_global2local[now_all_node[i]] = i
            node_index_local2global[i] = now_all_node[i]
        now_all_local_u = np.array(now_all_edge['u'].map(node_index_global2local))
        now_all_local_i = np.array(now_all_edge['i'].map(node_index_global2local))
        new_edge = now_all_edge[now_all_edge.ts>=now_time-tbatch_timespan]
        ###No new sample to evaluate
        if new_edge.shape[0] == 0:
            temporal_graph_dic[cache_key] = None
            return None, None, None, None, None, None, None
        ##Each new edge is a sample and if there are multiple same edge (u, i) in the new graph, only keep the latest one
        new_edge = new_edge.groupby(["u"]).tail(1)

        new_u_local = np.array(new_edge['u'].map(node_index_global2local))
        new_i_local = np.array(new_edge["i"].map(node_index_global2local))
        tmp_dic["induct_edge_lis_bool"] = None
        if state != "Train":
            induct_u_node =  new_edge['u'].isin(masked_u_node).to_numpy()
            induct_i_node = new_edge['i'].isin(masked_i_node).to_numpy()
            induct_edge_lis = induct_u_node | induct_i_node
            tmp_dic["induct_edge_lis_bool"] = induct_edge_lis
            #print(tmp_dic["induct_edge_lis_bool"].shape, new_edge.shape
        
        #add bidirection + add self-loop
        adj = spp.coo_matrix((np.ones(int(now_all_local_u.shape[0]+now_all_local_i.shape[0]+len(now_all_node))), (np.concatenate([now_all_local_u, now_all_local_i, list(range(len(now_all_node)))]), np.concatenate([now_all_local_i, now_all_local_u, list(range(len(now_all_node)))]))))
        now_graph = dgl.DGLGraph(adj)
        now_graph.readonly()
        tmp_dic["graph"] = now_graph
        tmp_dic["now_all_node"] = now_all_node
        tmp_dic["now_all_edge"] = np.array(now_all_edge.index)
        tmp_dic["now_dst_node_local"] = set(now_all_local_i.tolist())
        tmp_dic["t_now"] = now_time
        tmp_dic["new_src_local"] = new_u_local
        tmp_dic["new_dst_local"] = new_i_local
        tmp_dic["new_src_node_label"] = np.array(new_edge.label)
        temporal_graph_dic[cache_key] = tmp_dic
    if temporal_graph_dic[cache_key]:
        tmp_dic = temporal_graph_dic[cache_key]
        now_graph = copy.deepcopy(tmp_dic["graph"])
        now_graph.ndata["node_raw_feat"] = node_feature[tmp_dic["now_all_node"]]
        now_all_edge_feat = edge_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["edge_raw_feat"] = torch.cat([now_all_edge_feat, now_all_edge_feat, torch.zeros(len(tmp_dic["now_all_node"]), edge_feature.shape[-1])], dim=0)     ###undirected
        now_all_edge_t = timestamp_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["t"] = torch.cat([now_all_edge_t, now_all_edge_t, torch.zeros(len(tmp_dic["now_all_node"]), timestamp_feature.shape[-1])], dim=0)
        return now_graph, tmp_dic["new_src_local"], tmp_dic["new_dst_local"], tmp_dic["now_dst_node_local"], tmp_dic["new_src_node_label"], tmp_dic["induct_edge_lis_bool"], now_time
    else:
        return None, None, None, None, None, None, None



gnn_model = model_module.TGAT_nf(num_layers=args.n_layer, n_head=args.n_head, node_dim=node_feature.shape[-1], d_k=node_feature.shape[-1], d_v=node_feature.shape[-1], d_T=node_feature.shape[-1], edge_dim=node_feature.shape[-1], device=device, dropout=args.drop_out)

node_classifier = model_module.LR(node_feature.shape[-1])
optimizer = torch.optim.Adam(list(node_classifier.parameters())+list(gnn_model.parameters()), lr=args.lr)


pretrained_model = args.pretrained
if pretrained_model != "None":
    print("Load:", pretrained_model)
    gnn_model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))


criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])).to(device)
gnn_model.to(device)
node_classifier.to(device)
#gnn_model.eval()

###It seems that they change them to eval but not freeze
#for param in gnn_model.parameters():
#    param.requires_grad = False



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

###Link Prediction
## Build Graph Based on the Current Time
epoch_num = args.n_epoch

best_auc_roc = 0
best_auc_roc_epoch = 0
def run_model(start_timestamp, end_timestamp, state):
    global best_auc_roc, best_auc_roc_epoch
    all_score, all_label, all_loss = [], [], AverageMeter()
    tmp_tbatch_num = int((end_timestamp-start_timestamp)/tbatch_timespan)+1
    print_freq = tmp_tbatch_num // 3 + 1
    start_index = 0
    if state == "Train":
        is_train = True
        if "small" not in args.data: ##not in writing code mode
            start_index = 5 ##do not train with very first graph
    else:
        is_train = False


    tbatch_index = list(range(start_index, tmp_tbatch_num+1))
    if args.shuffle:
        random.shuffle(tbatch_index)
    start_time = time.time()

    if is_train:
        node_classifier.train()
        gnn_model.train()
    else:
        node_classifier.eval()
        gnn_model.eval

    for index, time_index in enumerate(tbatch_index):
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            now_graph, new_src_node_local, new_dst_node_local, now_dst_node_local, new_src_node_label, induct_edge_lis_bool, t_now = build_temporal_graph_cache(node_feature, linkage_df, time_index, start_timestamp, tbatch_timespan, state)
            if now_graph is None:
                continue
            label  = torch.Tensor(new_src_node_label).to(device)
            for nf in dgl.contrib.sampling.NeighborSampler(g=now_graph, batch_size=new_src_node_local.shape[0], expand_factor=args.expand_factor, neighbor_type='in', shuffle=False, num_hops=args.n_layer, seed_nodes=torch.Tensor(new_src_node_local).type(torch.int64), num_workers=1):
                nf.copy_from_parent(ctx=torch.device(device))
                optimizer.zero_grad()
                node_embedding = gnn_model(nf, t_now)[nf.map_from_parent_nid(layer_id=args.n_layer, parent_nids=torch.Tensor(new_src_node_local).type(torch.int64), remap_local=True),...]
                prob = node_classifier(node_embedding).squeeze()
                loss = criterion(prob, label)/args.pos_weight
                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        del now_graph
        torch.cuda.empty_cache()
        with torch.no_grad():
            all_score.append(prob.cpu().detach().numpy())
            all_label.append(label.cpu().detach().numpy())
            all_loss.update(loss.item(), label.shape[0])

    print("Time:", time.time()-start_time)
    roc_auc = roc_auc_score(np.concatenate(all_label), np.concatenate(all_score))
    if state == "Val":
        state = "**** " + state
        if best_auc_roc < roc_auc:
            best_auc_roc = roc_auc
            best_auc_roc_epoch = epoch
        print("Best Val, Epoch {0}, AUC {1:.4f} (Epoch {2})".format(epoch, best_auc_roc, best_auc_roc_epoch))
    if state == "Test":
        state = "!!!! " + state
    if state == "Train":
        print(all_loss.avg, roc_auc)
    print(state+' Epoch: [{0}][{1}/{2}][{all_loss.count}], AUC {roc_auc:.4f},  Loss {all_loss.val:.2e}({all_loss.avg:.2e})'.format(epoch, tmp_tbatch_num, tmp_tbatch_num, roc_auc=roc_auc, all_loss=all_loss))


for epoch in range(1, args.n_epoch+1):
    #Train
    run_model(start_timestamp = entire_start_timestamp, end_timestamp = val_start_timetamp, state = "Train")
    if epoch % args.val_interval == 0:
        #print("Eval Epoch %d"%(epoch))
        run_model(start_timestamp = val_start_timetamp, end_timestamp = test_start_timetamp, state = "Val")
    if epoch % args.test_interval == 0:
        #print("Test Epoch %d"%(epoch))
        run_model(start_timestamp = test_start_timetamp, end_timestamp=entire_end_timestamp, state ="Test")
    if epoch % args.snapshot_interval == 0:
        print("Epoch %d Save Model"%(epoch))
        file_path1 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_gnn.model")
        file_path2 = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_link_cls.model")
        if torch.cuda.is_available():
            torch.save(gnn_model.state_dict(), file_path1)
            torch.save(node_classifier.state_dict(), file_path2)
        else:
            torch.save(gnn_model.state_dict(), file_path1)
            torch.save(node_classifier.state_dict(), file_path2)