import torch
import torch.nn as nn

class F1Loss(nn.Module):
    """Differentiable F1 score loss.
    
    Input:
        pred (torch.Tensor): Model predictions [B, 1, H, W]
        target (torch.Tensor): Ground truth [B, 1, H, W]
    Returns:
        torch.Tensor: Scalar loss value [1]
    """
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate true positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        # Calculate precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        
        return 1 - f1  # Convert to loss 
