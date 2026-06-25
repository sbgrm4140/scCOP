import os
import argparse
import numpy as np
import h5py
import torch
import scanpy as sc
from time import time
from torch.utils.data import DataLoader, TensorDataset

# As config_utils is not in the current module, assuming it is in the parent level or PYTHONPATH
# If not, ensure it can be imported
try:
    from config_utils import load_config, get_param
except ImportError:
    print("Warning: config_utils not found. Ensure it is in your PYTHONPATH.")

from utils import filtering, normalize, find_intersection_with_positions
from model import COP
from annotation import CellAnnotator

def main():
    parser = argparse.ArgumentParser(description="parser for running scCOP")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path, e.g., config.yaml")
    parser.add_argument("--mode", type=str, choices=["full", "no_swap", "annotate"], default="full", 
                        help="Run mode: full (default, with swap), no_swap (ablation, without swap), or annotate (run cell annotation only)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # If annotation mode is selected, run annotation and exit
    if args.mode == "annotate":
        print("Running cell annotation...")
        annotator = CellAnnotator(config)
        annotator.run_annotation_pipeline()
        return

    # Get data related parameters
    batch_size = get_param(config, "data.batch_size")
    input_path = get_param(config, "data.input_path")
    input_file_name = get_param(config, "data.input_file_name")
    output_path = get_param(config, "data.output_path")

    # Get data key name parameters
    key_x1 = get_param(config, "data.key_x1")
    key_x2 = get_param(config, "data.key_x2")
    key_name1 = get_param(config, "data.key_name1")
    key_name2 = get_param(config, "data.key_name2")
    key_labels = get_param(config, "data.key_labels")

    # Get model parameters
    activation = get_param(config, "model.activation")
    sigma1 = get_param(config, "model.sigma1")
    sigma2 = get_param(config, "model.sigma2")

    # Get training parameters
    pretrain_lr = get_param(config, "training.pretrain_lr")
    clustering_lr = get_param(config, "training.clustering_lr")
    pretrain_epochs = get_param(config, "training.pretrain_epochs")
    clustering_max_iter = get_param(config, "training.clustering_max_iter")
    cutoff1 = get_param(config, "training.cutoff1")
    cutoff2 = get_param(config, "training.cutoff2")
    cutoff3 = get_param(config, "training.cutoff3") # Only used in full version
    update_interval = get_param(config, "training.update_interval")
    tol = get_param(config, "training.tol")

    # Get model hyperparameters
    alpha = get_param(config, "hyperparams.alpha")
    gamma = get_param(config, "hyperparams.gamma")
    tau = get_param(config, "hyperparams.tau")
    phi1 = get_param(config, "hyperparams.phi1")
    phi2 = get_param(config, "hyperparams.phi2")
    resolution = get_param(config, "hyperparams.resolution")

    # Get network structure parameters
    encodeLayer1 = get_param(config, "network.encodeLayer1")
    encodeLayer2 = get_param(config, "network.encodeLayer2")
    decodeLayer1 = get_param(config, "network.decodeLayer1")
    decodeLayer2 = get_param(config, "network.decodeLayer2")

    # Get device parameters
    device = get_param(config, "system.device")

    # Use paths and filenames from config file
    file_path = f"{input_path}/{input_file_name}.h5"
    data_mat = h5py.File(file_path, 'r')

    keys = list(data_mat.keys())
    name1 = np.array(data_mat[key_name1][:], dtype=np.str_)
    name2 = np.array(data_mat[key_name2][:], dtype=np.str_)
    x1 = np.array(data_mat[key_x1], dtype=np.float32)
    x2 = np.array(data_mat[key_x2], dtype=np.float32)
    if key_labels in keys:  
        y = np.array(data_mat[key_labels])
        print("y_nclusters: ", len(set(y)))
    else:
        y = None
    data_mat.close()

    print("x1.shape: ", x1.shape)
    print("x2.shape: ", x2.shape)
    
    # Data preprocessing
    barcodes = np.arange(x1.shape[0])
    if y is not None:
        x1, x2, name1, name2, barcodes, y = filtering(x1, x2, name1, name2, barcodes, Y=y)
    else:
        x1, x2, name1, name2, barcodes = filtering(x1, x2, name1, name2, barcodes, Y=None)

    adata1 = sc.AnnData(x1)
    adata1.var['name'] = name1
    adata1.obs['barcodes'] = barcodes
    if y is not None:
        adata1.obs['Group'] = y

    adata1 = normalize(adata1, size_factors=True, get_highly_val=True, normalize_input=True, logtrans_input=True)

    adata2 = sc.AnnData(x2) 
    adata2.var['name'] = name2
    adata2.obs['barcodes'] = barcodes
    if y is not None:
        adata2.obs['Group'] = y

    adata2 = normalize(adata2, size_factors=True, get_highly_val=True, normalize_input=True, logtrans_input=True)

    # Find the intersection of the two datasets
    index, positions1, positions2 = find_intersection_with_positions(adata1.obs['barcodes'], adata2.obs['barcodes'])
    adata1 = adata1[positions1, :]
    adata2 = adata2[positions2, :]
    if y is not None:
        y = np.array(adata1.obs['Group'])

    # Create new H5 file
    if not os.path.exists(f'{output_path}/{input_file_name}'):
        os.makedirs(f'{output_path}/{input_file_name}')
    filtered_file_path = f"{output_path}/{input_file_name}/{input_file_name}_filtered.h5"
    with h5py.File(filtered_file_path, 'w') as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('name1', data=adata1.var['name'], dtype=dt)
        f.create_dataset('name2', data=adata2.var['name'], dtype=dt)
        f.create_dataset('X1', data=adata1.raw.X, compression='gzip')
        f.create_dataset('X2', data=adata2.raw.X, compression='gzip')

    # Model training
    encodeLayer = list(map(int, encodeLayer1))
    decodeLayer1 = list(map(int, decodeLayer1))
    decodeLayer2 = list(map(int, decodeLayer2))

    # Load data
    x1_tensor = torch.Tensor(adata1.X).to(device)
    x2_tensor = torch.Tensor(adata2.X).to(device)
    X_raw1_tensor = torch.Tensor(adata1.raw.X).to(device)
    sf1_tensor = torch.Tensor(adata1.obs.size_factors).to(device)
    X_raw2_tensor = torch.Tensor(adata2.raw.X).to(device)
    sf2_tensor = torch.Tensor(adata2.obs.size_factors).to(device)

    dataset = TensorDataset(x1_tensor, X_raw1_tensor, sf1_tensor, x2_tensor, X_raw2_tensor, sf2_tensor)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    use_swap = (args.mode == "full")

    model = COP(input_dim1=adata1.X.shape[1], input_dim2=adata2.X.shape[1], tau=tau,
                encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                activation='gelu', sigma1=sigma1, sigma2=sigma2, alpha=alpha, n_prototype=batch_size, gamma=gamma, 
                phi1=phi1, phi2=phi2, device=device, use_swap=use_swap).to(device)

    t0 = time()
    model.pretrain_autoencoder(dataloader, lr=pretrain_lr, cutoff1=cutoff1, cutoff2=cutoff2, epochs=pretrain_epochs)
    print('Pretraining time: %d seconds.' % int(time() - t0))

    # Model training and evaluation
    if y is not None:
        n_clusters = len(set(y))
        acc, ami, nmi, ari = model.fit(X1=x1_tensor, X2=x2_tensor, dataloader=dataloader, cutoff3=cutoff3,
                                    n_clusters=n_clusters, num_epochs=clustering_max_iter, update_interval=update_interval, 
                                    tol=tol, lr=clustering_lr, y=y,
                                    adata1=adata1, adata2=adata2, output_path=output_path, input_file_name=input_file_name, batch_size=batch_size)
        if acc is not None:
            print(f"FinalMetrics: acc={acc:.5f}, ami={ami:.5f}, nmi={nmi:.5f}, ari={ari:.5f}")
    else:
        acc, ami, nmi, ari = model.fit(X1=x1_tensor, X2=x2_tensor, dataloader=dataloader, cutoff3=cutoff3,
                                    n_clusters=None, resolution=resolution, num_epochs=clustering_max_iter, update_interval=update_interval,
                                    tol=tol, lr=clustering_lr, y=None,
                                    adata1=adata1, adata2=adata2, output_path=output_path, input_file_name=input_file_name, batch_size=batch_size)

    print('Total time: %d seconds.' % int(time() - t0))

if __name__ == "__main__":
    main()
