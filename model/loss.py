import torch
import torch.nn as nn
import torch.nn.functional as F

class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        result = t1 + t2

        result = torch.mean(result)
        return result

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result

class SwappedPrediction(nn.Module):
    def __init__(self):
        super(SwappedPrediction, self).__init__()
    
    def forward(self, projection1, projection2, prototype, temperature=0.1):
        # Normalize prototype
        prototype = nn.functional.normalize(prototype, dim=1, p=2)
        
        # Calculate similarity scores
        scores1 = projection1 @ prototype.t() / temperature
        scores2 = projection2 @ prototype.t() / temperature
        
        # Calculate soft labels
        with torch.no_grad():
            q1 = self.Sinkhorn_Knopp(scores1.detach())
            q2 = self.Sinkhorn_Knopp(scores2.detach())
        
        # Calculate prediction probability
        p1 = F.softmax(scores1, dim=-1)
        p2 = F.softmax(scores2, dim=-1)
        
        # Cross entropy loss
        loss_12 = -torch.mean(torch.sum(q1 * torch.log(p2 + 1e-7), dim=1))
        loss_21 = -torch.mean(torch.sum(q2 * torch.log(p1 + 1e-7), dim=1))
        
        # Regularization term
        entropy_loss = -torch.mean(torch.sum(p1 * torch.log(p1 + 1e-7), dim=1)) - \
                      torch.mean(torch.sum(p2 * torch.log(p2 + 1e-7), dim=1))
                      
        # Total loss
        loss = 0.5 * (loss_12 + loss_21) + 0.1 * entropy_loss
        
        return loss

    def Sinkhorn_Knopp(self, scores, eps=1, niters=3):
        Q = torch.exp(scores / eps).T
        Q = Q / torch.sum(Q, dim=0)
        K, B = Q.shape
        
        r = torch.ones(K, device = scores.device) / K
        c = torch.ones(B, device = scores.device) / B
        for _ in range(niters):
            Q = Q * (r / torch.sum(Q, dim=1)).unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T
