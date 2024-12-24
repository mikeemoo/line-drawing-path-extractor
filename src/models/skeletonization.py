import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block for decoder"""
    def __init__(self, channels, kernel_size, padding, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class Skeletonization(nn.Module):
    """Neural network for extracting single-pixel-wide skeletons from line drawings.
    
    Input: [B, 1, H, W] - Batch of grayscale images
    Output: [B, 1, H, W] - Binary skeleton predictions
    where B = batch size, H, W = image dimensions
    """
    def __init__(self, dropout_rate=0.5, base_channels=64, 
                 decoder_kernel_size=3,
                 decoder_residual_blocks=0):
        super(Skeletonization, self).__init__()
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_residual_blocks = decoder_residual_blocks
        
        # Encoder with 4 layers instead of 5, smaller channel multipliers
        self.enc1 = self.conv_block(1, base_channels, dropout_rate, dilation=1)
        self.enc2 = self.conv_block(base_channels, base_channels*2, dropout_rate, dilation=1)
        self.enc3 = self.conv_block(base_channels*2, base_channels*3, dropout_rate, dilation=2)
        self.enc4 = self.conv_block(base_channels*3, base_channels*4, dropout_rate, dilation=2)

        # Decoder with 4 layers instead of 5
        self.dec4 = self.upconv_block(base_channels*4, base_channels*3, dropout_rate)
        self.dec3 = self.upconv_block(base_channels*6, base_channels*2, dropout_rate)  # Adjusted input channels
        self.dec2 = self.upconv_block(base_channels*4, base_channels, dropout_rate)    # Adjusted input channels
        self.dec1 = self.upconv_block(base_channels*2, base_channels, dropout_rate)    # Adjusted input channels

        self.skeleton_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, dropout_rate, dilation=1):
        """Conv block with optional dilation"""
        layers = []
        
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                        padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ])
        
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, dropout_rate):
        layers = []

        layers.extend([
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, out_channels)
        ])
        
        # Add residual blocks
        padding = (self.decoder_kernel_size - 1) // 2
        for _ in range(self.decoder_residual_blocks):
            layers.append(ResidualBlock(
                out_channels, 
                kernel_size=self.decoder_kernel_size, 
                padding=padding,
                dropout_rate=dropout_rate
            ))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([e3, d4], dim=1))
        d2 = self.dec2(torch.cat([e2, d3], dim=1))
        d1 = self.dec1(torch.cat([e1, d2], dim=1))
        
        # Final skeleton prediction
        skeleton_output = self.skeleton_head(d1)

        return skeleton_output 
