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
                 max_points=3,
                 start_edge_prob=0.5,
                 fixed_seed=None):
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
            fixed_seed: Seed for fixed data generation
        """
        self.size = size
        self.image_size = image_size
        self.noise_level = noise_level
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_points = min_points
        self.max_points = max_points
        self.start_edge_prob = start_edge_prob
        if fixed_seed is not None:
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Generate a single training sample."""
        # Create blank images
        input_img = np.zeros(self.image_size, dtype=np.float32)
        full_skeleton_img = np.zeros(self.image_size, dtype=np.float32)
        
        # Create PIL images for drawing
        pil_input = Image.new('L', self.image_size, 0)
        draw_input = ImageDraw.Draw(pil_input)
        
        # Create an array to track how many lines each pixel belongs to
        pixel_line_count = np.zeros(self.image_size, dtype=np.int32)
        
        # Generate random number of lines
        num_lines = random.randint(self.min_lines, self.max_lines)
        
        # Store thick line pixels for each skeleton line
        thick_pixels_per_line = []  # List of (thick_pixels, skeleton_points) tuples
        
        for _ in range(num_lines):
            points = random_points(
                self.image_size[0], 
                self.image_size[1],
                self.start_edge_prob,
                self.min_points,
                self.max_points
            )
            spline_points = catmull_rom_spline(points)
            
            # Create a separate PIL image for this thick line
            line_img = Image.new('L', self.image_size, 0)
            line_draw = ImageDraw.Draw(line_img)
            
            # Draw thick line
            thickness = random.randint(1, 5)
            draw_thick_line(draw_input, spline_points, thickness)
            draw_thick_line(line_draw, spline_points, thickness)
            
            # Get thick line pixels
            line_array = np.array(line_img)
            y_coords, x_coords = np.where(line_array >= 128)
            thick_pixels = set(zip(x_coords, y_coords))  # Store as (x,y)
            
            # Update pixel line count
            for x, y in thick_pixels:
                pixel_line_count[y, x] += 1
            
            # Generate skeleton points for this line
            skeleton_points = set()
            for i in range(len(spline_points) - 1):
                x1, y1 = map(int, spline_points[i])
                x2, y2 = map(int, spline_points[i + 1])
                skeleton_line = bresenham_line(x1, y1, x2, y2)
                skeleton_points.update(skeleton_line)
                
                # Add to full skeleton
                for x, y in skeleton_line:
                    if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                        full_skeleton_img[y, x] = 1.0
            
            thick_pixels_per_line.append((thick_pixels, skeleton_points))
        
        # Convert input image to numpy array and normalize
        input_img = np.array(pil_input, dtype=np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, input_img.shape)
        input_img = np.clip(input_img + noise, 0, 1)
        
        # Find valid pixels (those belonging to only one line)
        valid_pixels_per_line = []
        for thick_pixels, skeleton_points in thick_pixels_per_line:
            # Filter to keep only pixels that belong to one line
            valid_pixels = {(x, y) for x, y in thick_pixels 
                          if pixel_line_count[y, x] == 1}
            if valid_pixels:  # Only add if there are valid pixels
                valid_pixels_per_line.append((valid_pixels, skeleton_points))
        
        # If no valid pixels found, retry generation
        if not valid_pixels_per_line:
            return self.__getitem__(idx)
        
        # Randomly select one line that has valid pixels
        selected_line_idx = random.randint(0, len(valid_pixels_per_line) - 1)
        valid_pixels, skeleton_points = valid_pixels_per_line[selected_line_idx]
        
        # Randomly select one pixel from the valid pixels
        selected_pixel = random.choice(list(valid_pixels))
        
        # Create query point tensor (y, x)
        query_point = torch.tensor([selected_pixel[1], selected_pixel[0]], dtype=torch.long)
        
        # Create selected skeleton image
        selected_skeleton = np.zeros(self.image_size, dtype=np.float32)
        for x, y in skeleton_points:
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                selected_skeleton[y, x] = 1.0
        
        # Convert to tensors and add channel dimension
        input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
        selected_skeleton_tensor = torch.from_numpy(selected_skeleton).float().unsqueeze(0)
        full_skeleton_tensor = torch.from_numpy(full_skeleton_img).float().unsqueeze(0)
        
        # Add validation checks
        if selected_skeleton.sum() == 0:
            print(f"Warning: Generated skeleton at idx {idx} has no active pixels")
            return self.__getitem__(idx)  # Retry generation
        
        return input_tensor, query_point, selected_skeleton_tensor, full_skeleton_tensor

def create_dataloaders(batch_size=32, 
                      train_size=1000,
                      val_size=100,
                      num_workers=4,
                      **dataset_kwargs):
    """Create training and validation dataloaders."""
    train_dataset = LineDrawingDataset(
        size=train_size, 
        fixed_seed=None,  # Random training data
        **dataset_kwargs
    )
    
    val_dataset = LineDrawingDataset(
        size=val_size, 
        fixed_seed=42,  # Fixed seed for validation
        **dataset_kwargs
    )
    
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
