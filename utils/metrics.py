import numpy as np
import scanpy as sc
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    # ind is a tuple of arrays: (row_indices, col_indices) in newer scipy versions
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

def GetCluster(X, res, n_neighbors):
    adata0 = sc.AnnData(X)
    if adata0.shape[0] > 200000:
        np.random.seed(adata0.shape[0]) # set seed 
        adata0 = adata0[np.random.choice(adata0.shape[0], 200000, replace=False)] 
    sc.pp.neighbors(adata0, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.louvain(adata0, resolution=res)
    Y_pred_init = adata0.obs['louvain']
    Y_pred_init = np.asarray(Y_pred_init, dtype=int)
    if np.unique(Y_pred_init).shape[0] <= 1:  # avoid only a cluster
        exit("Error: There is only a cluster detected. The resolution: " + str(res) + " is too small, please choose a larger resolution!!")
    else: 
        print("Estimated n_clusters is: ", np.shape(np.unique(Y_pred_init))[0]) 
    return np.shape(np.unique(Y_pred_init))[0]
