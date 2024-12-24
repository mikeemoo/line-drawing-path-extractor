import torch
import torch.nn as nn

class SkeletonRepairNet(nn.Module):
    """Neural network for repairing damaged skeleton lines.
    
    Input: [B, 1, H, W] - Batch of damaged skeleton images
    Output: [B, 1, H, W] - Repaired skeleton predictions
    where B = batch size, H, W = image dimensions
    """
    def __init__(self, in_channels=1):
        super(SkeletonRepairNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # Middle layers with dilated convolutions for larger receptive field
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        
        # Final layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        
        # Middle
        middle = self.middle(enc2)
        
        # Decoder with skip connections
        dec2 = self.dec2(torch.cat([middle, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        
        # Final layer with sigmoid for binary output
        out = torch.sigmoid(self.final(dec1))
        
        return out
