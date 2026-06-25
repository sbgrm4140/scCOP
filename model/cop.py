import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn import metrics

from utils.metrics import cluster_acc, GetCluster
from .layers import MeanAct, DispAct, buildNetwork
from .loss import ZINBLoss, SwappedPrediction

class COP(nn.Module):
    def __init__(self, input_dim1, input_dim2,
            encodeLayer, decodeLayer1, decodeLayer2, tau, device,
            activation, sigma1, sigma2, alpha, n_prototype, gamma, phi1, phi2, use_swap=True):
        super(COP, self).__init__()
        self.tau = tau
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.device = device
        self.use_swap = use_swap

        # Encoders
        self.encoder1 = buildNetwork([input_dim1] + encodeLayer, activation)
        self.encoder2 = buildNetwork([input_dim2] + encodeLayer, activation)
        self.decoder1 = buildNetwork(decodeLayer1, activation=activation)
        self.decoder2 = buildNetwork(decodeLayer2, activation=activation)       
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.z_dim = encodeLayer[-1]

        # projector for swap prediction (Keep structure for compatibility whether swap is used or not)
        self.projector = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_prototype)
        )
        if self.use_swap:
            self.prototype = Parameter(torch.normal(0, 1, [n_prototype, n_prototype]), requires_grad=True)
        else:
            self.prototype = None

        # Add components for marker gene identification
        self.marker_attention1 = nn.Sequential(
            nn.Linear(self.z_dim, input_dim1),
            nn.Sigmoid()
        )
        self.marker_attention2 = nn.Sequential(
            nn.Linear(self.z_dim, input_dim2),
            nn.Sigmoid()
        )
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def kmeans_loss(self, z):
        dist1 = self.tau * torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)
    
    def forward_encode(self, x1, x2):
        z1_clean = self.encoder1(x1)
        z2_clean = self.encoder2(x2)
        z_clean = torch.cat([z1_clean, z2_clean], dim=1)
        return z_clean
    
    def forward_swap(self, x1, x2):
        if random.random() < 0.5:
            x1_noisy = x1 + torch.randn_like(x1) * self.sigma1
            x2_noisy = x2 + torch.randn_like(x2) * self.sigma2
        else:
            noise_mask1 = (torch.rand(x1.size(0), device=x1.device) < 0.5).float().view(-1, 1)
            noise_mask2 = (torch.rand(x2.size(0), device=x2.device) < 0.5).float().view(-1, 1)

            noise1 = torch.randn_like(x1) * self.sigma1
            noise2 = torch.randn_like(x2) * self.sigma2

            x1_noisy = x1 + noise1 * noise_mask1
            x2_noisy = x2 + noise2 * noise_mask2
        
        z1_clean = self.encoder1(x1)
        z2_clean = self.encoder2(x2)
        z1_noisy = self.encoder1(x1_noisy)
        z2_noisy = self.encoder2(x2_noisy)
        
        z_clean = torch.cat([z1_clean, z2_clean], dim=1)
        
        z_num, lq = self.cal_latent(z_clean)
        
        h1 = self.decoder1(z1_noisy)
        _mean1 = self.dec_mean1(h1)
        _disp1 = self.dec_disp1(h1)
        _pi1 = self.dec_pi1(h1)
        
        h2 = self.decoder2(z2_noisy)
        _mean2 = self.dec_mean2(h2)
        _disp2 = self.dec_disp2(h2)
        _pi2 = self.dec_pi2(h2)

        if self.use_swap:
            projection_a = F.normalize(self.projector(z1_clean), dim=1)
            projection_b = F.normalize(self.projector(z2_clean), dim=1)
            return z_clean, z_num, lq, _mean1, _mean2, _disp1, _disp2, _pi1, _pi2, projection_a, projection_b, self.prototype
        else:
            return z_clean, z_num, lq, _mean1, _mean2, _disp1, _disp2, _pi1, _pi2, None, None, None

    def encodeBatch(self, X1, X2, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        encoded = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch)
            inputs2 = Variable(x2batch)
            z = self.forward_encode(inputs1, inputs2)
            encoded.append(z.data)
        encoded = torch.cat(encoded, dim=0)
        return encoded
    
    def get_marker_genes(self, z, x1, x2, cluster_assignments):
        gene_weights1 = self.marker_attention1(z[:, :self.z_dim])  
        gene_weights2 = self.marker_attention2(z[:, self.z_dim:])  
        
        markers_dict1 = {}
        markers_dict2 = {}
        scores_dict1 = {}  
        scores_dict2 = {}  
        
        for cluster_id in torch.unique(cluster_assignments):
            cluster_mask = (cluster_assignments == cluster_id)
            
            cluster_weights1 = gene_weights1[cluster_mask].mean(dim=0)
            cluster_weights2 = gene_weights2[cluster_mask].mean(dim=0)
            
            cluster_expr1 = x1[cluster_mask].mean(dim=0)
            cluster_expr2 = x2[cluster_mask].mean(dim=0)
            
            marker_score1 = cluster_weights1 * cluster_expr1
            marker_score2 = cluster_weights2 * cluster_expr2
            
            top_markers1 = torch.topk(marker_score1, k=min(50, marker_score1.size(0)))
            top_markers2 = torch.topk(marker_score2, k=min(50, marker_score2.size(0)))
            
            markers_dict1[cluster_id.item()] = top_markers1.indices.cpu().numpy()
            markers_dict2[cluster_id.item()] = top_markers2.indices.cpu().numpy()
            scores_dict1[cluster_id.item()] = top_markers1.values.cpu().numpy()  
            scores_dict2[cluster_id.item()] = top_markers2.values.cpu().numpy()  
            
        return markers_dict1, markers_dict2, scores_dict1, scores_dict2
    
    def calculate_marker_loss(self, z, x1, x2, cluster_assignments):
        gene_weights1 = self.marker_attention1(z[:, :self.z_dim])
        gene_weights2 = self.marker_attention2(z[:, self.z_dim:])
        
        loss = 0
        unique_clusters = torch.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_assignments == cluster_id)
            other_mask = ~cluster_mask
            
            if torch.sum(cluster_mask) == 0 or torch.sum(other_mask) == 0:
                continue
            
            cluster_expr1 = x1[cluster_mask].mean(dim=0)
            other_expr1 = x1[other_mask].mean(dim=0)
            cluster_expr2 = x2[cluster_mask].mean(dim=0)
            other_expr2 = x2[other_mask].mean(dim=0)
            
            diff1 = torch.abs(cluster_expr1 - other_expr1)
            diff2 = torch.abs(cluster_expr2 - other_expr2)
            
            loss += torch.mean((1 - gene_weights1) * diff1)
            loss += torch.mean((1 - gene_weights2) * diff2)
            
            loss += 0.1 * (torch.mean(torch.abs(gene_weights1)) + torch.mean(torch.abs(gene_weights2)))
            
            loss += 0.01 * (torch.mean(torch.abs(gene_weights1[:-1] - gene_weights1[1:])) + 
                        torch.mean(torch.abs(gene_weights2[:-1] - gene_weights2[1:])))
        
        return loss / max(len(unique_clusters), 1)
    
    def pretrain_autoencoder(self, dataloader, lr, cutoff1, cutoff2, epochs):
        swap_loss_fn = SwappedPrediction() if self.use_swap else None
        print("Pretraining stage. Mode: ", "Full (with swap)" if self.use_swap else "Ablation (no swap)")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            recon_loss1_val = 0
            recon_loss2_val = 0
            kl_loss_val = 0
            swap_loss_val_acc = 0

            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor, projection_1, projection_2, prototype = self.forward_swap(x1_batch, x2_batch)
                recon_loss1 = self.zinb_loss(x=x_raw1_batch, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_batch)
                recon_loss2 = self.zinb_loss(x=x_raw2_batch, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_batch)
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) 

                if self.use_swap:
                    swap_loss_val = swap_loss_fn(projection_1, projection_2, prototype, temperature=1)
                else:
                    swap_loss_val = torch.tensor(0.0).to(self.device)

                if epoch+1 >= epochs * cutoff1 and epoch+1 < epochs * cutoff2:
                    loss = recon_loss1 + recon_loss2 + (swap_loss_val * self.phi2 if self.use_swap else 0)
                elif epoch+1 >= epochs * cutoff2:
                    loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1 + (swap_loss_val * self.phi2 if self.use_swap else 0)
                else:
                    loss = recon_loss1 + recon_loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = x1_batch.size(0)
                loss_val += loss.item() * batch_size
                recon_loss1_val += recon_loss1.item() * batch_size
                recon_loss2_val += recon_loss2.item() * batch_size
                
                if epoch+1 >= epochs * cutoff1 and epoch+1 < epochs * cutoff2:
                    if self.use_swap: swap_loss_val_acc += swap_loss_val.item() * batch_size
                elif epoch+1 >= epochs * cutoff2:
                    if self.use_swap: swap_loss_val_acc += swap_loss_val.item() * batch_size
                    kl_loss_val += kl_loss.item() * batch_size
            
            num_samples = len(dataloader.dataset)
            loss_val /= num_samples
            recon_loss1_val /= num_samples
            recon_loss2_val /= num_samples
            kl_loss_val /= num_samples
            swap_loss_val_acc /= num_samples
            if epoch%10 == 0:
               print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, ZINB loss2:{:.6f}, KL loss:{:.6f}, Swap loss:{:.6f}'.format(epoch+1, loss_val, recon_loss1_val, recon_loss2_val, kl_loss_val, swap_loss_val_acc))


    def fit(self, X1, X2, dataloader, cutoff3, lr=1., n_clusters=None, resolution=None, num_epochs=10, update_interval=1, tol=1e-3, y=None, adata1=None, adata2=None, output_path=None, input_file_name=None, batch_size=256):
        swap_loss_fn = SwappedPrediction() if self.use_swap else None
        Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
        if n_clusters is None:
            with torch.no_grad():
                Zdata0 = Zdata.cpu().numpy()
                n_clusters = GetCluster(Zdata0, resolution, n_neighbors=40)
        self.mu = Parameter(torch.Tensor(n_clusters, 2*self.z_dim), requires_grad=True)
        
        main_optimizer = optim.Adadelta(
            [p for n, p in self.named_parameters() if not n.startswith('marker_attention')], 
            lr=lr, 
            rho=.95
        )
        marker_optimizer = optim.Adadelta(
            [p for n, p in self.named_parameters() if n.startswith('marker_attention')], 
            lr=lr, 
            rho=.95
        )

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, AMI= %.4f, NMI= %.4f, ARI= %.4f' % (acc, ami, nmi, ari))
        
        self.train()
        
        final_acc, final_ami, final_nmi, final_ari, final_epoch = 0, 0, 0, 0, 0
        epoch_tol = 0 

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
                dist, _ = self.kmeans_loss(Zdata)
                self.y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
                if y is not None:
                    final_acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                    final_nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    final_epoch = epoch+1
                    print('Clustering   %d: ACC= %.4f, AMI= %.4f, NMI= %.4f, ARI= %.4f,  n_clusters= %.0f' % (epoch+1, final_acc, final_ami, final_nmi, final_ari, len(set(self.y_pred))))

                num_samples = len(dataloader.dataset)
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num_samples
                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    epoch_tol += 1
                    if cutoff3 is not None and epoch+1 > epoch_tol/cutoff3:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                   
            loss_val = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            swap_loss_val_acc = 0.0
            marker_loss_val = 0.0

            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor, projection_1, projection_2, prototype = self.forward_swap(x1_batch, x2_batch)
                dist, cluster_loss = self.kmeans_loss(zbatch)
                recon_loss1 = self.zinb_loss(x=x_raw1_batch, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_batch)
                recon_loss2 = self.zinb_loss(x=x_raw2_batch, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_batch)
                
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch)

                if self.use_swap:
                    swap_loss_val = swap_loss_fn(projection_1, projection_2, prototype, temperature=1)
                else:
                    swap_loss_val = torch.tensor(0.0).to(self.device)

                main_loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1 + cluster_loss * self.gamma + (swap_loss_val * self.phi2 if self.use_swap else 0)

                main_optimizer.zero_grad()
                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                main_optimizer.step()

                if (cutoff3 is not None and epoch+1 >= num_epochs * cutoff3) or delta_label < tol:
                    batch_cluster_assignments = torch.argmin(dist.detach(), dim=1)
                    marker_loss = self.calculate_marker_loss(
                        zbatch.detach(), 
                        x1_batch, 
                        x2_batch, 
                        batch_cluster_assignments
                    )
                    marker_optimizer.zero_grad()
                    marker_loss.backward()
                    marker_optimizer.step()
                    marker_loss_val += marker_loss.data * len(x1_batch)
                else: 
                    marker_loss_val = 0.0

                cluster_loss_val += cluster_loss.data * len(x1_batch)
                recon_loss1_val += recon_loss1.data * len(x1_batch)
                recon_loss2_val += recon_loss2.data * len(x2_batch)
                kl_loss_val += kl_loss.data * len(x1_batch)
                if self.use_swap:
                    swap_loss_val_acc += swap_loss_val.data * len(x1_batch)
                
                loss_val = cluster_loss_val + recon_loss1_val + recon_loss2_val + kl_loss_val + swap_loss_val_acc + marker_loss_val

            if epoch%10 == 0:
               num_samples = len(dataloader.dataset)
               print("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss1: %.6f ZINB Loss2: %.6f KL Loss: %.6f Swap Loss: %.6f Marker Loss: %.6f" % (
                     epoch + 1, loss_val / num_samples, cluster_loss_val / num_samples, recon_loss1_val / num_samples, recon_loss2_val / num_samples, kl_loss_val / num_samples, swap_loss_val_acc / num_samples, marker_loss_val / num_samples))

        # Output logic matching original script
        if output_path and input_file_name and adata1 is not None and adata2 is not None:
            z_all = self.encodeBatch(X1, X2, batch_size=batch_size)
            dist, _ = self.kmeans_loss(z_all)
            y_pred = torch.argmin(dist, dim=1)
            
            with torch.no_grad():
                markers_dict1, markers_dict2, scores_dict1, scores_dict2 = self.get_marker_genes(z_all, X1, X2, y_pred)
                
                markers_df1 = pd.DataFrame()
                markers_df2 = pd.DataFrame()
                scores_df1 = pd.DataFrame()  
                scores_df2 = pd.DataFrame()  

                cluster_pred = pd.DataFrame()
                cluster_pred['barcode'] = adata1.obs.index
                cluster_pred['cluster'] = y_pred.cpu().numpy()

                z_df = pd.DataFrame(z_all.cpu().numpy(), index=None, columns=None)

                max_markers = max(len(genes) for genes in markers_dict1.values()) if len(markers_dict1) > 0 else 0
                
                for cluster_id in markers_dict1.keys():
                    genes1 = list(markers_dict1[cluster_id])
                    genes2 = list(markers_dict2[cluster_id])
                    scores1 = list(scores_dict1[cluster_id])
                    scores2 = list(scores_dict2[cluster_id])
                    
                    genes1_names = [adata1.var.name.iloc[i] for i in genes1] + [None] * (max_markers - len(genes1))
                    genes2_names = [adata2.var.name.iloc[i] for i in genes2] + [None] * (max_markers - len(genes2))
                    
                    scores1_pad = list(scores1) + [None] * (max_markers - len(scores1))
                    scores2_pad = list(scores2) + [None] * (max_markers - len(scores2))
                    
                    markers_df1[f'Cluster_{cluster_id}'] = genes1_names
                    markers_df2[f'Cluster_{cluster_id}'] = genes2_names
                    scores_df1[f'Cluster_{cluster_id}'] = scores1_pad
                    scores_df2[f'Cluster_{cluster_id}'] = scores2_pad

                out_dir = f'{output_path}/{input_file_name}'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                markers_df1.to_csv(f'{out_dir}/markers_modality1.csv', index=False)
                markers_df2.to_csv(f'{out_dir}/markers_modality2.csv', index=False)
                scores_df1.to_csv(f'{out_dir}/markers_scores_modality1.csv', index=False)
                scores_df2.to_csv(f'{out_dir}/markers_scores_modality2.csv', index=False)
                cluster_pred.to_csv(f'{out_dir}/cluster_pred.csv', index=False)
                z_df.to_csv(f'{out_dir}/z.csv', index=False)

        if y is not None:
            return final_acc, final_ami, final_nmi, final_ari
        else:
            return None, None, None, None
