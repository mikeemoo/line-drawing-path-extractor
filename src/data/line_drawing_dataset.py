import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import random

from ..utils.data_generation import (
    random_points,
    catmull_rom_spline,
    bresenham_line,
    draw_thick_line
)

class LineDrawingDataset(Dataset):
    """Dataset for line drawing skeletonization."""
    def __init__(self, 
                 size=1000, 
                 image_size=(64, 64), 
                 noise_level=0.1,
                 min_lines=1,
                 max_lines=3,
                 min_points=3,
                 max_points=5,
                 start_edge_prob=0.5,
                 overdraw_prob=0.05,  # Probability of a pixel being considered for overdraw
                 pixel_remove_prob=0.1):  # Probability of removing a pixel from the skeleton
        """
        Args:
            size: Number of samples in dataset
            image_size: Tuple of (height, width)
            noise_level: Amount of Gaussian noise to add
            min_lines: Minimum number of lines per image
            max_lines: Maximum number of lines per image
            min_points: Minimum number of points per line
            max_points: Maximum number of points per line
            start_edge_prob: Probability of starting a line from an edge
            overdraw_prob: Probability of a pixel being considered for overdraw
            pixel_remove_prob: Probability of removing a pixel from the skeleton
        """
        self.size = size
        self.image_size = image_size
        self.noise_level = noise_level
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_points = min_points
        self.max_points = max_points
        self.start_edge_prob = start_edge_prob
        self.overdraw_prob = overdraw_prob
        self.pixel_remove_prob = pixel_remove_prob

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Generate a single training sample.
        
        Returns:
            tuple: Contains:
                - input_tensor (torch.Tensor): Input image [1, H, W]
                - skeleton_tensor (torch.Tensor): Ground truth skeleton [1, H, W]
                - damaged_tensor (torch.Tensor): Damaged skeleton for repair [1, H, W]
            where H, W = self.image_size (default: 64, 64)
        """
        # Create blank images
        input_img = np.zeros(self.image_size, dtype=np.float32)
        skeleton_img = np.zeros(self.image_size, dtype=np.float32)
        damaged_img = np.zeros(self.image_size, dtype=np.float32)
        
        # Create PIL images for drawing
        pil_input = Image.new('L', self.image_size, 0)
        draw_input = ImageDraw.Draw(pil_input)
        
        # Generate random number of lines
        num_lines = random.randint(self.min_lines, self.max_lines)
        
        all_skeleton_points = set()  # Use a set to avoid duplicates
        
        for _ in range(num_lines):
            points = random_points(
                self.image_size[0], 
                self.image_size[1],
                self.start_edge_prob,
                self.min_points,
                self.max_points
            )
            spline_points = catmull_rom_spline(points)
            
            # Draw thick line for input
            thickness = random.randint(1, 5)
            draw_thick_line(draw_input, spline_points, thickness)
            
            # Generate skeleton points using Bresenham's algorithm
            for i in range(len(spline_points) - 1):
                x1, y1 = map(int, spline_points[i])
                x2, y2 = map(int, spline_points[i + 1])
                skeleton_line = bresenham_line(x1, y1, x2, y2)
                all_skeleton_points.update(skeleton_line)
        
        # Convert input image to numpy array and normalize
        input_img = np.array(pil_input, dtype=np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, input_img.shape)
        input_img = np.clip(input_img + noise, 0, 1)
        
        # Create skeleton image and damaged image
        for x, y in all_skeleton_points:
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                skeleton_img[y, x] = 1.0
                # Add skeleton points to damaged with random removal
                if random.random() >= self.pixel_remove_prob:
                    damaged_img[y, x] = 1.0
        
        # Add subtle overdrawing
        overdrawn_points = set()
        for x, y in all_skeleton_points:
            # Only consider this pixel for overdrawing with some probability
            if random.random() < self.overdraw_prob:
                # Choose one random adjacent pixel (4-connected neighborhood)
                dx = random.choice([0, 1, 0, -1])
                dy = random.choice([1, 0, -1, 0]) if dx == 0 else 0
                
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.image_size[0] and 
                    0 <= new_y < self.image_size[1] and
                    (new_x, new_y) not in all_skeleton_points):
                    overdrawn_points.add((new_x, new_y))
        
        # Add overdrawn points to damaged image
        for x, y in overdrawn_points:
            damaged_img[y, x] = 1.0
        
        # Convert to tensors and add channel dimension
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        skeleton_tensor = torch.from_numpy(skeleton_img).float().unsqueeze(0)
        damaged_tensor = torch.from_numpy(damaged_img).float().unsqueeze(0)
        
        return input_tensor, skeleton_tensor, damaged_tensor

def create_dataloaders(batch_size=32, 
                      train_size=1000,
                      val_size=100,
                      num_workers=4,
                      **dataset_kwargs):
    """Create training and validation dataloaders."""
    train_dataset = LineDrawingDataset(size=train_size, **dataset_kwargs)
    val_dataset = LineDrawingDataset(size=val_size, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 
