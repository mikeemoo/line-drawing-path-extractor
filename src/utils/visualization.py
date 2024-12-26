import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Set the backend before importing pyplot

def visualize_all_predictions(input_image, pixel_mask, predicted_path, ground_truth_path, save_path=None):
    """Visualize model predictions for path extraction.
    
    Args:
        input_image (torch.Tensor): Input image [1, H, W]
        pixel_mask (torch.Tensor): Single pixel mask [1, H, W]
        predicted_path (torch.Tensor): Predicted skeleton path [1, H, W]
        ground_truth_path (torch.Tensor): Ground truth skeleton path [1, H, W]
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure containing the visualization
    """
    # Convert tensors to numpy arrays and remove channel dimension
    input_image = input_image.squeeze().numpy()
    pixel_mask = pixel_mask.squeeze().numpy()
    predicted_path = predicted_path.squeeze().numpy()
    ground_truth_path = ground_truth_path.squeeze().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Path Extraction Results', fontsize=16)
    
    # Plot input image
    axes[0, 0].imshow(input_image, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Plot pixel mask
    axes[0, 1].imshow(input_image, cmap='gray')
    mask_y, mask_x = np.where(pixel_mask > 0.5)
    if len(mask_y) > 0:  # If there are points to plot
        axes[0, 1].scatter(mask_x, mask_y, c='red', s=50)
    axes[0, 1].set_title('Selected Pixel')
    axes[0, 1].axis('off')
    
    # Plot predicted path
    axes[1, 0].imshow(predicted_path, cmap='gray')
    axes[1, 0].set_title('Predicted Path')
    axes[1, 0].axis('off')
    
    # Plot ground truth path
    axes[1, 1].imshow(ground_truth_path, cmap='gray')
    axes[1, 1].set_title('Ground Truth Path')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    return fig 
