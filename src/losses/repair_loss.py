import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonRepairLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(SkeletonRepairLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Binary Cross Entropy
        bce_loss = F.binary_cross_entropy(pred, target)
        
        # Structural similarity term using convolution to check neighborhood
        kernel = torch.ones(1, 1, 3, 3).to(pred.device)
        pred_neighb = F.conv2d(pred, kernel, padding=1)
        target_neighb = F.conv2d(target, kernel, padding=1)
        struct_loss = F.mse_loss(pred_neighb, target_neighb)
        
        return self.alpha * bce_loss + (1 - self.alpha) * struct_loss
