import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Constants moved to dataset class, only keeping drawing-related ones here
MIN_POINTS = 3
MAX_POINTS = 5

def random_points(width, height, start_edge_prob=0.5, min_points=MIN_POINTS, max_points=MAX_POINTS):
    """Generate random points for a line, optionally starting from an edge.
    
    Args:
        width: Image width
        height: Image height
        start_edge_prob: Probability of starting from an edge
        min_points: Minimum number of points in the line
        max_points: Maximum number of points in the line
    """
    num_points = random.randint(min_points, max_points)
    points = []
    
    if random.random() < start_edge_prob:
        if random.random() < 0.5:
            # Vertical edges
            x = random.choice([0, width-1])
            y = random.randint(0, height-1)
        else:
            # Horizontal edges
            x = random.randint(0, width-1)
            y = random.choice([0, height-1])
    else:
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
    points.append((x, y))
    
    for _ in range(num_points - 1):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        points.append((x, y))
    return points

def catmull_rom_spline(points, num_samples=10):
    """Generate a smooth curve through the given points using Catmull-Rom spline.
    
    Args:
        points: List of (x, y) tuples defining control points
        num_samples (int): Number of points to generate between each pair
    
    Returns:
        List[Tuple[float, float]]: List of (x, y) points along the spline
    """
    if len(points) < 2:
        return points
    
    curve_points = []
    extended_points = [points[0]] + points + [points[-1]]
    for i in range(len(points)-1):
        p0, p1, p2, p3 = extended_points[i], extended_points[i+1], extended_points[i+2], extended_points[i+3]
        for t_i in range(num_samples):
            t = t_i / num_samples
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t**2 +
                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t**3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t**2 +
                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t**3
            )
            curve_points.append((x, y))
    if points:
        curve_points.append(points[-1])
    return curve_points

def bresenham_line(x0, y0, x1, y1):
    """Generate points for a line using Bresenham's algorithm.
    
    Args:
        x0, y0: Starting point coordinates
        x1, y1: Ending point coordinates
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x, y))
    return points

def draw_thick_line(draw, points, thickness):
    """Draw an anti-aliased line with specified thickness.
    
    Args:
        draw: PIL ImageDraw object
        points: List of (x, y) tuples defining the line path
        thickness (int): Line thickness in pixels
    
    Note: Operates on the image in-place. The image dimensions determine
    the coordinate space for the points.
    """
    scale = 4  # Scale factor for anti-aliasing
    large_size = (draw.im.size[0] * scale, draw.im.size[1] * scale)
    large_img = Image.new('L', large_size, 0)
    large_draw = ImageDraw.Draw(large_img)
    
    # Scale up the points and thickness
    scaled_points = [(x * scale, y * scale) for x, y in points]
    scaled_thickness = thickness * scale
    
    # Draw smooth line segments
    for i in range(len(scaled_points) - 1):
        x1, y1 = scaled_points[i]
        x2, y2 = scaled_points[i + 1]
        large_draw.line([(x1, y1), (x2, y2)], fill=255, width=scaled_thickness + 2)
    
    # Apply slight Gaussian blur for smoother edges
    large_img = large_img.filter(ImageFilter.GaussianBlur(radius=scale/2))
    
    # Resize back down with high-quality anti-aliasing
    small_img = large_img.resize(draw.im.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays and blend
    current = np.array(draw.im, dtype=np.uint8).reshape(draw.im.size[::-1])
    new_line = np.array(small_img, dtype=np.uint8)
    blended = np.maximum(current, new_line)
    
    # Update the original image
    draw.im.putdata(Image.fromarray(blended, 'L').getdata()) 
