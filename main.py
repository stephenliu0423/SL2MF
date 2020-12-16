import pdb
import time
import numpy as np
from lmf import LMF
from functions import mean_confidence_interval
from functions import load_data
from collections import defaultdict
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from functions import evalution_bal


inter_pairs, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, pro_id_mapping = load_data()
# num_nodes = 6375
num_nodes = np.max(inter_pairs) + 1

x, y = np.triu_indices(num_nodes, k=1)
c_set = set(zip(x, y)) - set(zip(inter_pairs[:, 0], inter_pairs[:, 1])) - set(zip(inter_pairs[:, 1], inter_pairs[:, 0]))
noninter_pairs = np.array(list(c_set))

reorder = np.arange(len(inter_pairs))
np.random.shuffle(reorder)
inter_pairs = inter_pairs[reorder]

reorder_neg = np.arange(len(noninter_pairs))
np.random.shuffle(reorder_neg)
noninter_pairs = noninter_pairs[reorder_neg]
print("Data downloaded!")

go_sim_mat = go_sim_mat.toarray()
go_sim_mat = go_sim_mat+go_sim_mat.T

cv_flag = True
prng = np.random.RandomState(123)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
num = len(pro_id_mapping.keys())
pos_edge_kf = kf.split(inter_pairs)
neg_edge_kf = kf.split(noninter_pairs)

for nn_size in [50]:
    auc_pro, aupr_pro = [], []
    auc_pair, aupr_pair = [], []
    t = time.time()
    for train, test in pos_edge_kf:
        neg_train, neg_test = next(neg_edge_kf)
        model = LMF(num_factors=150, nn_size=nn_size, theta=2.0**(-5), reg=10.0**(-2), alpha=1*10.0**(0),
                    beta=1*10.0**(0), beta1=1*10.0**(0), beta2=1*10.0**(0), max_iter=300)
        print(str(model))
        x, y = inter_pairs[train, 0], inter_pairs[train, 1]
        IntMat = np.zeros((num, num))
        W = np.ones((num, num))
        IntMat[x, y] = 1
        IntMat[y, x] = 1
        W[x, y] = 50
        W[y, x] = W[x, y]

        model.fix(IntMat, W, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat)
        # auc_val, aupr_val, metric = evalution_bal(np.dot(model.U, model.U.T), inter_pairs[test, :], noninter_pairs[neg_test, :])
        auc_val, aupr_val = evalution_bal(np.dot(model.U, model.U.T), inter_pairs[test, :],
                                                  noninter_pairs[neg_test, :])
        auc_pair.append(auc_val)
        aupr_pair.append(aupr_val)
        print("metrics over protein pairs: auc %f, aupr %f, time: %f\n" % (auc_val, aupr_val, time.time()-t))

    m1, sdv1 = mean_confidence_interval(auc_pair)
    m2, sdv2 = mean_confidence_interval(aupr_pair)
    print("Average metrics over pairs: auc_mean:%s, auc_sdv:%s, aupr_mean:%s, aupr_sdv:%s\n" %(m1, sdv1, m2, sdv2))
