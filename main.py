
import time
import argparse
import numpy as np
from lmf import LMF
from functions import mean_confidence_interval
from functions import load_data
from collections import defaultdict
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from functions import evalution_bal


def main(args):
    inter_pairs, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, pro_id_mapping = load_data()
    num_nodes = np.max(inter_pairs) + 1

    x, y = np.triu_indices(num_nodes, k=1)
    c_set = set(zip(x, y)) - set(zip(inter_pairs[:, 0], inter_pairs[:, 1])) - set(zip(inter_pairs[:, 1], inter_pairs[:, 0]))
    noninter_pairs = np.array(list(c_set))

    reorder = np.arange(len(inter_pairs))
    prng = np.random.RandomState(args.seed)
    prng.shuffle(reorder)
    inter_pairs = inter_pairs[reorder]

    reorder_neg = np.arange(len(noninter_pairs))
    prng.shuffle(reorder_neg)

    noninter_pairs = noninter_pairs[reorder_neg]
    print("Data downloaded!")

    go_sim_mat = go_sim_mat.toarray()
    go_sim_mat = go_sim_mat+go_sim_mat.T

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=prng)

    num = len(pro_id_mapping.keys())
    pos_edge_kf = kf.split(inter_pairs)
    neg_edge_kf = kf.split(noninter_pairs)

    # auc_pro, aupr_pro = [], []
    auc_pair, aupr_pair = [], []
    t = time.time()
    for train, test in pos_edge_kf:
        neg_train, neg_test = next(neg_edge_kf)
        model = LMF(num_factors=args.num_factors, nn_size=args.nn_size, theta=args.theta, reg=args.reg,
                    alpha=args.alpha, beta=args.beta, beta1=args.beta, beta2=args.beta,
                    max_iter=args.max_iter, seed=args.seed)
        print(str(model))
        x, y = inter_pairs[train, 0], inter_pairs[train, 1]
        IntMat = np.zeros((num, num))
        W = np.ones((num, num))
        IntMat[x, y] = 1
        IntMat[y, x] = 1
        W[x, y] = args.weight
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The SL2MF algorithm")
    parser.add_argument("--nn_size", default=50, help="the size of the neighborhood")
    parser.add_argument("--num_factors", default=50, help="the dimension of embedding vector")
    parser.add_argument("--theta", default=2**(-5), help="the learning rate")
    parser.add_argument("--reg", default=0.01)
    parser.add_argument("--alpha", default=1.0)
    parser.add_argument("--beta", default=1.0)
    parser.add_argument("--max_iter", default=300)
    parser.add_argument("--seed", default=123, help="random seed")
    parser.add_argument("--cv_folds", default=5, help="the number of cross-validation folds")
    parser.add_argument("--weight", default=50, help="the weight used to balance the positive and unlabelled data" )
    args = parser.parse_args()
    main(args)
