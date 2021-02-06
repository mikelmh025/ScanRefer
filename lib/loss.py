import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, dim_in=1):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=dim_in)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=dim_in).mean()

        return loss