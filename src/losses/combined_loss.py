
from src.losses.f1_loss import F1Loss
import torch.nn as nn
import torch

class CombinedLoss(nn.Module):
    """Combines F1 and MSE loss with scaling.
    
    MSE is scaled up to be comparable to F1 loss:
    - F1 loss typically ranges from 0 to 1
    - MSE is typically much smaller (e.g., 0.009)
    - Scale factor of ~50 makes them comparable
    """
    def __init__(self, mse_scale=50.0, f1_weight=0.5, epsilon=1e-7):
        super(CombinedLoss, self).__init__()
        self.mse_scale = mse_scale
        self.f1_weight = f1_weight
        self.mse_loss = nn.MSELoss()
        self.f1_loss = F1Loss(epsilon=epsilon)
        
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target) * self.mse_scale
        f1 = self.f1_loss(pred, target)
        
        # Optionally log individual losses
        if torch.is_grad_enabled():  # Only during training
            print(f"\rMSE: {mse.item():.4f}, F1: {f1.item():.4f}", end="")
            
        return (self.f1_weight * f1) + ((1 - self.f1_weight) * mse) 
