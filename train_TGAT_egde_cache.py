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
import scipy.sparse as spp


model_file = "TGAT"
model_module = importlib.import_module("model_folder."+model_file)

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
parser.add_argument('--name', type=str, default="TGAT-L", help='The name of this setting')
parser.add_argument('--tbatch_num', type=int, default=3, help='tbatch_num')
parser.add_argument('--val_interval', type=int, default=1, help='every number of epoches to evaluate')
parser.add_argument('--test_interval', type=int, default=1, help='every number of epoches to test')
parser.add_argument('--snapshot_interval', type=int, default=5, help='every number of epoches to save snapshot of model')
parser.add_argument('--neg_sampling_ratio', type=int, default=1, help='The ratio of negative sampling')
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
#linkage_df["feat"] = edge_feature.tolist()
node_feature = torch.Tensor(node_feature).to(device)
edge_feature = torch.Tensor(edge_feature).to(device)
timestamp_feature = torch.Tensor(linkage_df.ts).unsqueeze(-1).to(device)

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



# # # def build_temporal_graph(node_feature, edge_list, time_index, start_timestamp, tbatch_timespan):
# # #     now_time = start_timestamp + (time_index+1) * tbatch_timespan
# # #     now_all_edge = edge_list[edge_list.ts<=now_time].copy()
# # #     now_src_node = list(now_all_edge.u.unique())
# # #     now_dst_node = list(now_all_edge.i.unique())
# # #     now_all_node = now_src_node + now_dst_node
# # #     now_all_node.sort()
# # #     node_index_global2local = {}
# # #     node_index_local2global = {}
# # #     for i in range(len(now_all_node)):
# # #         node_index_global2local[now_all_node[i]] = i
# # #         node_index_local2global[i] = now_all_node[i]
# # #     now_all_edge["local_u"] = now_all_edge['u'].map(node_index_global2local)
# # #     now_all_edge["local_i"] = now_all_edge['i'].map(node_index_global2local)

# # #     new_edge = now_all_edge[now_all_edge.ts>=now_time-tbatch_timespan]
# # #     ###No new sample to evaluate
# # #     if new_edge.shape[0] == 0:
# # #         return None, None, None, None, None, None
# # #     ##Each new edge is the a sample and if there are multiple same edge (u, i) in the new graph, only keep the latest one
# # #     new_edge = new_edge.groupby(["u",'i']).tail(1)
# # #     now_all_local_u = np.array(now_all_edge.local_u)
# # #     now_all_local_i = np.array(now_all_edge.local_i)
# # #     #undirected
    
# # #     adj = spp.coo_matrix((np.ones(int(now_all_local_u.shape[0]*2)), (np.concatenate([now_all_local_u, now_all_local_i]), np.concatenate([now_all_local_i, now_all_local_u]))))
# # #     now_graph = dgl.DGLGraph(adj)
# # #     now_graph.ndata["node_raw_feat"] = torch.Tensor(node_feature[now_all_node])
# # #     now_all_edge_feat = np.stack(now_all_edge.feat.values)
# # #     now_all_edge_feat = np.concatenate([now_all_edge_feat, now_all_edge_feat], axis=0)

# # #     now_graph.edata["edge_raw_feat"] = torch.Tensor(now_all_edge_feat)
# # #     now_graph.edata["t"] = torch.Tensor(np.concatenate([now_all_edge.ts, now_all_edge.ts])).unsqueeze(-1)
# # #     return now_graph, new_edge, now_src_node, now_dst_node, node_index_global2local, now_time


###With cache
temporal_graph_dic = {}
def build_temporal_graph_cache(node_feature, edge_list, time_index, start_timestamp, tbatch_timespan):
    now_time = start_timestamp + (time_index+1) * tbatch_timespan
    if time_index not in temporal_graph_dic:
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
            temporal_graph_dic[time_index] = None
            return None, None, None, None, None, None
        ##Each new edge is the a sample and if there are multiple same edge (u, i) in the new graph, only keep the latest one
        new_edge = new_edge.groupby(["u"]).tail(1)
        new_u_local = np.array(new_edge['u'].map(node_index_global2local))
        new_i_local = np.array(new_edge["i"].map(node_index_global2local))


        adj = spp.coo_matrix((np.ones(int(now_all_local_u.shape[0]*2)), (np.concatenate([now_all_local_u, now_all_local_i]), np.concatenate([now_all_local_i, now_all_local_u]))))
        now_graph = dgl.DGLGraph(adj)
        now_graph.readonly()
        tmp_dic["graph"] = now_graph
        tmp_dic["now_all_node"] = now_all_node
        #tmp_dic["now_src_node"] = now_src_node
        #tmp_dic["now_dst_node"] = now_dst_node
        tmp_dic["now_all_edge"] = np.array(now_all_edge.index)
        tmp_dic["now_dst_node_local"] = set(now_all_local_i.tolist())
        tmp_dic["t_now"] = now_time
        tmp_dic["new_src_local"] = new_u_local
        tmp_dic["new_dst_local"] = new_i_local
        tmp_dic["new_src_node_label"] = np.array(new_edge.label)
        temporal_graph_dic[time_index] = tmp_dic
    if temporal_graph_dic[time_index]:
        tmp_dic = temporal_graph_dic[time_index]
        now_graph = copy.deepcopy(tmp_dic["graph"])
        now_graph.ndata["node_raw_feat"] = node_feature[tmp_dic["now_all_node"]]
        now_all_edge_feat = edge_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["edge_raw_feat"] = torch.cat([now_all_edge_feat, now_all_edge_feat], dim=0)     ###undirected
        now_all_edge_t = timestamp_feature[tmp_dic["now_all_edge"]]
        now_graph.edata["t"] = torch.cat([now_all_edge_t, now_all_edge_t], dim=0)
        return now_graph, tmp_dic["new_src_local"], tmp_dic["new_dst_local"], tmp_dic["now_dst_node_local"], tmp_dic["new_src_node_label"], now_time
    else:
        return None, None, None, None, None, None

neg_sampling_ratio = args.neg_sampling_ratio


gnn_model = model_module.TGAT(num_layers=args.n_layer, n_head=args.n_head, node_dim=node_feature.shape[-1], d_k=node_feature.shape[-1], d_v=node_feature.shape[-1], d_T=node_feature.shape[-1], edge_dim=node_feature.shape[-1], device=device, dropout=args.drop_out)

link_classifier = model_module.MergeLayer(node_feature.shape[-1], node_feature.shape[-1], node_feature.shape[-1], 1)
optimizer = torch.optim.Adam(list(gnn_model.parameters())+list(link_classifier.parameters()), lr=args.lr)

criterion = torch.nn.BCELoss().to(device)
gnn_model.to(device)
link_classifier.to(device)



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

best_val_ap = 0
best_val_ap_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0
def run_model(start_timestamp, end_timestamp, state):
    global best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch
    time_index = 0
    all_acc, all_ap, all_f1, all_auc, all_m_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    tmp_tbatch_num = int((end_timestamp-start_timestamp)/tbatch_timespan)+1
    print_freq = tmp_tbatch_num // 3 + 1
    if state == "Train":
        is_train = True
        #time_index = tmp_tbatch_num//2
    else:
        is_train = False

    while start_timestamp + time_index * tbatch_timespan <= end_timestamp:
        start_time = time.time()
        #print(start_timestamp, end_timestamp, time_index, time_index * tbatch_timespan)
        now_graph, new_src_node_local, new_dst_node_local, now_dst_node_local, new_src_node_label, t_now = build_temporal_graph_cache(node_feature, linkage_df, time_index, start_timestamp, tbatch_timespan)
        time_index += 1
        if now_graph is None:
            print(time_index)
            continue
        now_graph.to(torch.device(device))
        
        optimizer.zero_grad()
        if is_train:
            gnn_model.train()
            link_classifier.train()
        else:
            gnn_model.eval()
            link_classifier.eval()
        with torch.set_grad_enabled(is_train):
            node_embedding = gnn_model(now_graph, t_now)

            del now_graph
            with torch.no_grad():
                ##dim: num_of_new_edge
                neg_dst_node = np.array(np.array([random.sample(now_dst_node_local-set([new_dst_node_local[i]]), neg_sampling_ratio) for i in range(len(new_dst_node_local))]))
                sample_src_node = np.repeat(new_src_node_local, neg_sampling_ratio+1, axis=-1).ravel()
                sample_dst_node = np.concatenate([np.expand_dims(new_dst_node_local, axis=-1), neg_dst_node], axis=-1).ravel()
                label = torch.Tensor(([1]+[0]*neg_sampling_ratio)*new_src_node_local.shape[0]).to(device)
            
            prob = link_classifier(node_embedding[sample_src_node], node_embedding[sample_dst_node]).sigmoid().squeeze()
            loss = criterion(prob, label)
            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            label = label.cpu().detach().numpy()
            pred_score = prob.cpu().detach().numpy()
            pred_label = pred_score > 0.5
            all_ap.update(average_precision_score(label, pred_score), label.shape[0])
            all_acc.update((pred_label==label).mean(), label.shape[0])
            all_m_loss.update(loss.item(), label.shape[0])
            all_auc.update(roc_auc_score(label, pred_score), label.shape[0])
            all_f1.update(f1_score(label, pred_label), label.shape[0])
        if is_train & (time_index % print_freq == 0):
            print('Train Epoch: [{0}][{1}/{2}][{all_ap.count}({3})], AP {all_ap.val:.4f}({all_ap.avg:.4f}), Acc {all_acc.val:.4f}({all_acc.avg:.4f}), Loss {all_m_loss.val:.2e}({all_m_loss.avg:.2e}), Auc {all_auc.val:.4f}({all_auc.avg:.4f}), F1 {all_f1.val:.4f}({all_f1.avg:.4f})'.format(epoch, time_index, tmp_tbatch_num, new_src_node_local.shape[0], all_ap=all_ap, all_acc=all_acc, all_m_loss=all_m_loss, all_auc=all_auc, all_f1=all_f1))
    print("Time:", time.time()-start_time)
    if state == "Val":
        state = "**** " + state
        if best_val_acc < all_acc.avg:
            best_val_acc = all_acc.avg
            best_val_acc_epoch = epoch
        if best_val_ap < all_ap.avg:
            best_val_ap = all_ap.avg
            best_val_ap_epoch = epoch
        print("Best Val, Epoch {0}, Acc {1:.4f} (Epoch {2}), AP {3:.4f} (Epoch {4})".format(epoch, best_val_acc, best_val_acc_epoch, best_val_ap, best_val_ap_epoch))
    if state == "Test":
        state = "!!!! " + state
    print(state+' Epoch: [{0}][{1}/{2}][{all_ap.count}], AP {all_ap.val:.4f}({all_ap.avg:.4f}), Acc {all_acc.val:.4f}({all_acc.avg:.4f}), Loss {all_m_loss.val:.2e}({all_m_loss.avg:.2e}), Auc {all_auc.val:.4f}({all_auc.avg:.4f}), F1 {all_f1.val:.4f}({all_f1.avg:.4f})'.format(epoch, time_index, tmp_tbatch_num, all_ap=all_ap, all_acc=all_acc, all_m_loss=all_m_loss, all_auc=all_auc, all_f1=all_f1))


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
            torch.save(link_classifier.state_dict(), file_path2)
        else:
            torch.save(gnn_model.state_dict(), file_path1)
            torch.save(link_classifier.state_dict(), file_path2)