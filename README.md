# Temporal-Graph

Implementation of TGAT in DGL.

Thanks to TGAT's authors' code: https://drive.google.com/drive/folders/1GaH8vusCXJj4ucayfO-PyHpnNsJRkB78?usp=sharing

Put raw data file in the "processed" folder first.

preprocess.py --  generate the data file we use.

train_TGAT_edge_nf.py -- TGAT for link prediction

train_TGAT_node_nf.py -- TGAT for node classification (no updating for GNN as in the TGAT's paper) 

train_TGAT_node_nf_sup.py -- TGAT for node classification (supervised)

train_TGAT_RNN.py -- combine TGAT an RNN
