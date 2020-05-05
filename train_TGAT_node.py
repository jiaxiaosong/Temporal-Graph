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
import os,sys
import pickle
import copy
import shutil
import importlib
from sklearn.metrics import roc_auc_score


model_file = "TGAT"
model_module = importlib.import_module("model_folder."+model_file)

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
parser.add_argument('--name', type=str, default="TGAT-N", help='The name of this setting')
parser.add_argument('--pretrained', type=str, default="None", help='The position of pretrained model')
parser.add_argument('--tbatch_num', type=int, default=3, help='tbatch_num')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
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
setting_name = args.name
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
linkage_df["feat"] = edge_feature.tolist()


### Train Val Test 0.7 - 0.15 - 0.15
entire_start_timestamp = float(linkage_df.ts.min())
val_start_timetamp, test_start_timetamp = list(np.quantile(linkage_df.ts, [0.70, 0.85]))
entire_end_timestamp = float(linkage_df.ts.max())

train_edge = linkage_df[linkage_df.ts<val_start_timetamp]
val_edge =  linkage_df[(linkage_df.ts>val_start_timetamp) & (linkage_df.ts<test_start_timetamp)]
test_edge = linkage_df[linkage_df.ts>test_start_timetamp]

###Different from the TGAT paper, we use the model in t-batch instead of only one loss for each item-node in one of the three sets. It is the same evaluation method as in the original dataset paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
tbatch_num = args.tbatch_num#500.0
tbatch_timespan = float((train_edge.ts.max()-train_edge.ts.min())/float(tbatch_num))

def build_temporal_graph(node_feature, edge_list, time_index, start_timestamp, tbatch_timespan):
    now_time = start_timestamp + (time_index+1) * tbatch_timespan
    now_all_edge = edge_list[edge_list.ts<=now_time].copy()
    now_src_node = list(now_all_edge.u.unique())
    now_tgt_node = list(now_all_edge.i.unique())
    now_all_node = now_src_node + now_tgt_node
    now_all_node.sort()
    node_index_global2local = {}
    node_index_local2global = {}
    for i in range(len(now_all_node)):
        node_index_global2local[now_all_node[i]] = i
        node_index_local2global[i] = now_all_node[i]
    now_all_edge["local_u"] = now_all_edge['u'].map(node_index_global2local)
    now_all_edge["local_i"] = now_all_edge['i'].map(node_index_global2local)

    new_edge = now_all_edge[now_all_edge.ts>=now_time-tbatch_timespan]
    ###No new sample to evaluate
    if new_edge.shape[0] == 0:
        return None, None, None, None, None, None
    ##Each new edge is the a sample and if there are multiple same edge (u, i) in the new graph, only keep the latest one
    new_edge = new_edge.groupby(["u",'i']).tail(1)

    now_graph = dgl.DGLGraph()
    now_graph.add_nodes(len(now_all_node))
    now_graph.ndata["node_raw_feat"] = torch.Tensor(node_feature[now_all_node]) 
    now_graph.add_edges(torch.Tensor(now_all_edge.local_u).type(torch.int64), torch.Tensor(now_all_edge.local_i).type(torch.int64), data={"t":torch.Tensor(now_all_edge.ts).unsqueeze(-1), "edge_raw_feat":torch.Tensor(now_all_edge.feat), "global_edge_index":torch.Tensor(now_all_edge.index)})

    ##Undirected-graph
    now_graph.add_edges(torch.Tensor(now_all_edge.local_i).type(torch.int64), torch.Tensor(now_all_edge.local_u).type(torch.int64), data={"t":torch.Tensor(now_all_edge.ts).unsqueeze(-1), "edge_raw_feat":torch.Tensor(now_all_edge.feat), "global_edge_index":torch.Tensor(now_all_edge.index)})
    return now_graph, new_edge, now_src_node, now_tgt_node, node_index_global2local, now_time



device = args.device#"cuda:0"#"cpu"


gnn_model = model_module.TGAT(num_layers=args.n_layer, n_head=args.n_head, node_dim=node_feature.shape[-1], d_k=node_feature.shape[-1], d_v=node_feature.shape[-1], d_T=node_feature.shape[-1], edge_dim=node_feature.shape[-1], device=device, dropout=args.drop_out)

node_classifier = model_module.LR(node_feature.shape[-1])
optimizer = torch.optim.Adam(node_classifier.parameters(), lr=args.lr)


pretrained_model = args.pretrained
if pretrained_model != "None":
    print("Load:", pretrained_model)
    gnn_model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))

criterion = torch.nn.BCELoss().to(device)
gnn_model.to(device)
node_classifier.to(device)
gnn_model.eval()
for param in gnn_model.parameters():
    param.requires_grad = False



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
    time_index = 0
    all_score, all_label, all_loss = [], [], AverageMeter()
    tmp_tbatch_num = int((end_timestamp-start_timestamp)/tbatch_timespan)+1
    print_freq = tmp_tbatch_num // 3 + 1
    if state == "Train":
        is_train = True
        #time_index = tmp_tbatch_num//2
    else:
        is_train = False

    while start_timestamp + time_index * tbatch_timespan <= end_timestamp:
        with torch.no_grad():
            start_time = time.time()
            now_graph, new_edge, now_src_node, now_dst_node, node_index_global2local, t_now = build_temporal_graph(node_feature, linkage_df, time_index, start_timestamp, tbatch_timespan)
            time_index += 1
            if new_edge is None:
                continue
            now_graph.to(torch.device(device))

            optimizer.zero_grad()
            node_embedding = gnn_model(now_graph, t_now)
            now_graph.clear()
            src_node = new_edge.local_u.to_numpy()
            label  = torch.Tensor(new_edge.label.to_numpy()).to(device)

        if is_train:
            node_classifier.train()
        else:
            node_classifier.eval()
        prob = node_classifier(node_embedding[src_node]).sigmoid().squeeze()
        loss = criterion(prob, label)
        if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            #label = label.cpu().detach().numpy()
            #pred_score = prob.cpu().detach().numpy()
            #all_loss.append(loss.item())
            all_score.append(prob.cpu().detach().numpy())
            all_label.append(label.cpu().detach().numpy())
            #all_auc_roc.update(roc_auc_score(label, pred_score), src_node.shape[0])
            all_loss.update(loss.item(), src_node.shape[0])
        #if is_train & (time_index % print_freq == 0):
            #print('Train Epoch: [{0}][{1}/{2}][{all_ap.count}({3})], AUC {all_auc_roc.val:.4f}({all_auc_roc.avg:.4f}), Loss {all_loss.val:.2e}({all_loss.avg:.2e})'.format(epoch, time_index, tmp_tbatch_num, new_edge.shape[0], all_auc_roc=all_auc_roc, all_loss=all_loss))
    #print("Time:", time.time()-start_time)
    roc_auc = roc_auc_score(np.concatenate(all_label), np.concatenate(all_score))
    #print((np.concatenate(all_label)==(np.concatenate(all_score)>0.5)).mean(), "1111111111111111111111111111111")
    if state == "Val":
        state = "**** " + state
        if best_auc_roc < roc_auc:
            best_auc_roc = roc_auc
            best_auc_roc_epoch = epoch
        #print("Best Val, Epoch {0}, AUC {1:.4f} (Epoch {2})".format(epoch, best_auc_roc, best_auc_roc_epoch))
    if state == "Test":
        state = "!!!! " + state
    if state == "Train":
        print(all_loss.avg, roc_auc)
    #print(all_loss.val)
    print(state+' Epoch: [{0}][{1}/{2}][{all_loss.count}], AUC {roc_auc:.4f},  Loss {all_loss.val:.2e}({all_loss.avg:.2e})'.format(epoch, time_index, tmp_tbatch_num, roc_auc=roc_auc, all_loss=all_loss))


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