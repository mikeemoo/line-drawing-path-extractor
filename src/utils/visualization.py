import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Set the backend before importing pyplot

def visualize_all_predictions(input_image, ground_truth, raw_pred):
    """Visualize all prediction methods side by side."""
    # Create a figure with 4 subplots in a single row
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    
    # Threshold predictions for visualization
    threshold = 0.5
    raw_pred_binary = (raw_pred > threshold).astype(float)
    
    # Plot each image
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input Image')
    
    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth')
    
    axes[2].imshow(raw_pred_binary, cmap='gray')
    axes[2].set_title('Model Prediction')
    
    # Create comparison visualization
    axes[3].imshow(ground_truth, cmap='gray')  # Show ground truth in grayscale
    # Add prediction overlay in red
    overlay = np.zeros((*ground_truth.shape, 4))  # RGBA
    overlay[..., 0] = raw_pred_binary  # Red channel for predictions
    overlay[..., 3] = raw_pred_binary * 0.7  # Alpha channel
    axes[3].imshow(overlay, cmap='Reds', alpha=0.5)
    axes[3].set_title('Prediction vs Ground Truth')
    
    # Turn off axes for all subplots
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig 
