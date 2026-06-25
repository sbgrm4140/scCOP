import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

def buildNetwork(layers, activation):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "selu":
            net.append(nn.SELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
        elif activation == "gelu":
            net.append(nn.GELU())
        elif activation == "softplus":
            net.append(nn.Softplus())
        elif activation is None:
            continue
    return nn.Sequential(*net)
