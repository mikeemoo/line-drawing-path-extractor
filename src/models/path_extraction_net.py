import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        # Make residual blocks more expressive
        self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=3, 
                              padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels*2)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm([channels])
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Reshape for attention
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Self-attention
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out + x_flat)  # Residual
        
        # Reshape back
        return attn_out.transpose(1, 2).reshape(b, c, h, w)

class EnhancedSkipBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels*2, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x, skip):
        combined = torch.cat([x, skip], dim=1)
        return self.norm(self.conv1(combined))

class PathExtractionNet(nn.Module):
    """Network for extracting a single skeleton line given an input image and query point.
    
    Input shapes:
        x: Input image tensor [B, 1, H, W]
            B: batch size
            1: single channel (grayscale)
            H, W: height and width (default: 64, 64)
        
        query_point: Selected pixel coordinates [B, 2]
            B: batch size
            2: (y, x) coordinates for each image in range [0, H-1] and [0, W-1]
    
    Output shape:
        Predicted skeleton line [B, 1, H, W]
            B: batch size
            1: single channel (probability map)
            H, W: same as input dimensions
    """
    def __init__(self, in_channels=2, hidden_channels=64):
        """
        Args:
            in_channels: Number of input channels (2 = image + query mask)
            hidden_channels: Number of channels in hidden layers (multiple of 8 for efficiency)
        """
        super().__init__()
        
        # Ensure hidden_channels is a multiple of 8 for efficiency
        hidden_channels = ((hidden_channels + 7) // 8) * 8
        
        # Make the network wider at the start
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Add more dilated blocks with different rates
        self.encoder = nn.ModuleList([
            ResidualBlock(hidden_channels, dilation=1),
            ResidualBlock(hidden_channels, dilation=2),
            ResidualBlock(hidden_channels, dilation=4)
        ])
        
        # Global context block with reduced attention heads for larger hidden_channels
        self.context = nn.Sequential(
            AttentionBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        
        # Match decoder to encoder
        self.decoder = nn.ModuleList([
            ResidualBlock(hidden_channels, dilation=4),
            ResidualBlock(hidden_channels, dilation=2),
            ResidualBlock(hidden_channels, dilation=1)
        ])
        
        # Final prediction with gradual channel reduction
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels//2, hidden_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(),
            nn.Conv2d(hidden_channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add dropout to the model
        self.dropout = nn.Dropout(0.1)
        
        # Add skip processors
        self.skip_processors = nn.ModuleList([
            EnhancedSkipBlock(hidden_channels)
            for _ in range(5)  # Changed to match number of encoder/decoder blocks
        ])
        
    def forward(self, x, query_point):
        """
        Args:
            x: Input image tensor [B, 1, H, W]
            query_point: Selected pixel coordinates [B, 2] in range [0, H-1] and [0, W-1]
        
        Returns:
            Predicted skeleton line probability map [B, 1, H, W]
            Values are in range [0, 1] due to sigmoid activation
        """
        # Store original input
        input_tensor = x.float()  # Convert to float32 at the start
        
        batch_size, _, h, w = input_tensor.shape
        
        # Create Gaussian mask around query point
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=input_tensor.device),
            torch.arange(w, device=input_tensor.device),
            indexing='ij'
        )
        y_grid = y_grid.float()
        x_grid = x_grid.float()
        
        mask = torch.zeros((batch_size, 1, h, w), device=input_tensor.device, dtype=torch.float32)
        
        for i in range(batch_size):
            y, x = query_point[i, 0], query_point[i, 1]
            dist = ((y_grid - y.float())**2 + (x_grid - x.float())**2) / 2.0  # Smaller spread
            mask[i, 0] = torch.exp(-dist)
            
        # Try making tensors contiguous before concatenation
        input_tensor = input_tensor.contiguous()
        mask = mask.contiguous()
        
        # Concatenate using input_tensor instead of x
        x = torch.cat([input_tensor, mask], dim=1)
        
        # Initial conv -> [B, hidden_channels, H, W]
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder:
            x = self.dropout(block(x))  # Add dropout
            skip_connections.append(x)
            
        # Global context -> [B, hidden_channels, H, W]
        x = self.context(x)
        
        # Decoder with enhanced skip connections (no multi-scale fusion)
        for block, skip in zip(self.decoder, reversed(skip_connections)):
            x = block(x + skip)  # Simple addition instead of enhanced processing
        
        # Final prediction -> [B, 1, H, W]
        return self.final(x)
