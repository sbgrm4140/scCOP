import scanpy as sc
import anndata as ad
import numpy as np

def filter_genes(X, min_cells=0):
    a = X != 0
    b = np.sum(a, axis=0) >= min_cells
    return b

def filter_cells(X, min_genes=0):
    a = X != 0
    b = np.sum(a, axis=1) >= min_genes
    return b

def filtering(X1, X2, name1, name2, barcodes, Y=None):
    genes_index1 = filter_genes(X1)
    genes_index2 = filter_genes(X2)
    X1 = X1[:, genes_index1]
    X2 = X2[:, genes_index2]
    name1 = name1[genes_index1]
    name2 = name2[genes_index2]

    cells_index1 = filter_cells(X1)
    cells_index2 = filter_cells(X2)
    index = list(cells_index1) and list(cells_index2)
    X1 = X1[index, :]
    X2 = X2[index, :]
    barcodes = barcodes[index]
    if Y is not None:
        Y = Y[index]
        a = Y[0]
        if isinstance(a, np.int_) or isinstance(a, np.int8) or isinstance(a, np.int32) or isinstance(a, np.int64) or isinstance(a, int):
            Y = Y 
        elif isinstance(a, np.float_) or isinstance(a, np.float16) or isinstance(a, np.float32) or isinstance(a, np.float64) or isinstance(a, float):
            Y = [int(i) for i in Y]
        elif isinstance(a, np.str_) or isinstance(a, str):
            c = []
            for i in Y:
                if i not in c:
                    c.append(i)
            num = list(np.arange(len(c)))
            yy = np.arange(len(Y))
            for i in range(len(c)):
                idx = Y == c[i]
                yy[idx] = num[i] 
            Y = yy
        else:
            raise ValueError("Please check Y again")
        return X1, X2, name1, name2, barcodes, Y
    return X1, X2, name1, name2, barcodes


def normalize(adata, get_highly_val=True, size_factors=True, normalize_input=True, logtrans_input=True):
    n_top_genes = 2000
    if get_highly_val:
        adata2 = adata.copy()
        sc.pp.log1p(adata2)
        sc.pp.highly_variable_genes(adata2, n_top_genes=n_top_genes) # Identify highly variable genes
        adata = adata[:, adata2.var.highly_variable] # Filter
    adata.raw = adata.copy()
    
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)
    
    return adata


def find_intersection_with_positions(list1, list2):
    # Use sets for fast intersection
    intersection = set(list1) & set(list2)
    
    # Mark positions of intersection elements in both lists
    positions1 = [i for i, x in enumerate(list1) if x in intersection]
    positions2 = [i for i, x in enumerate(list2) if x in intersection]
    
    return list(intersection), positions1, positions2
