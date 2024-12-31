import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from ..data.line_drawing_dataset import create_dataloaders
from .visualization import visualize_all_predictions

class EarlyStopping:
    """Standard early stopping based on loss."""
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

def calculate_metrics(skeleton_pred, skeleton_target):
    """Calculate various metrics for model evaluation.
    
    Args:
        skeleton_pred (torch.Tensor): Model predictions [B, 1, H, W]
        skeleton_target (torch.Tensor): Ground truth [B, 1, H, W]
    
    Returns:
        dict: Contains metrics:
            - 'iou' (float): Intersection over Union
            - 'precision' (float): Precision score
            - 'recall' (float): Recall score
            - 'f1_score' (float): F1 score
    """
    # Convert predictions to binary using threshold
    skeleton_binary = (skeleton_pred > 0.5).float()
    
    # Calculate IoU for skeleton
    intersection = (skeleton_binary * skeleton_target).sum()
    union = skeleton_binary.sum() + skeleton_target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    # Calculate precision and recall
    true_positives = (skeleton_binary * skeleton_target).sum()
    false_positives = (skeleton_binary * (1 - skeleton_target)).sum()
    false_negatives = ((1 - skeleton_binary) * skeleton_target).sum()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
   
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }

def train_model(model, num_epochs=200, batch_size=4, learning_rate=0.004, 
                gradient_clip_value=1.0, criterion=None, hidden_channels=64, run_name=None):
    """Training loop for path extraction model.
    
    The model takes an input image and a query point to extract the corresponding skeleton path:
    Input: (image [B, 1, H, W], query_point [B, 2]) -> skeleton path [B, 1, H, W]
    
    Args:
        model: The PathExtractionNet model instance
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        gradient_clip_value: Value to clip gradients at
        criterion: Loss function to use
        hidden_channels: Number of channels in hidden layers (multiple of 8 for efficiency)
        run_name: Optional name for this training run (used in TensorBoard)
    """
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Create unique run name with timestamp
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = f'runs/path_extraction_{timestamp}'
    if run_name:
        run_dir += f'_{run_name}'
    
    # Initialize TensorBoard writer with unique run name
    writer = SummaryWriter(run_dir)
    logging.info(f"TensorBoard logs will be saved to: {run_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on: {device}")
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'context'):
        model.context.requires_grad_(True)
        for m in model.context.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.use_checkpoint = True
                if hasattr(m, 'num_heads') and m.num_heads > 2:
                    m.num_heads = 2
                logging.debug("Enabled gradient checkpointing for attention layers")
    
    # Move model to device
    model = model.to(device)
    torch.cuda.empty_cache()  # Clear any memory before starting
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.005)
    
    # Log model parameters
    writer.add_scalar('Parameters/Total', sum(p.numel() for p in model.parameters()))
    
    # Create dataloaders with smaller batch sizes
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        train_size=500,  # Further reduced dataset size
        val_size=50,
        num_workers=0  # Disable multiprocessing to save memory
    )
    logging.info(f"Created dataloaders with batch size {batch_size}")
    
    # Enable automatic mixed precision with more aggressive optimization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    scaler = torch.cuda.amp.GradScaler()
    logging.info("Enabled automatic mixed precision training with TF32")
    
    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()  # Clear cache at start of epoch
        
        training_loss = 0
        epoch_start_time = perf_counter()
        
        # Add gradient norm monitoring
        total_grad_norm = 0
        num_layers = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                num_layers += 1
        
        avg_grad_norm = total_grad_norm / num_layers if num_layers > 0 else 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, query_points, targets, _) in enumerate(progress_bar):
            # Move data to device just before use
            inputs = inputs.to(device)
            query_points = query_points.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):  # Force FP16
                output = model(inputs, query_points)
                batch_loss = criterion(output, targets)
            
            # Scale loss and call backward
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
            
            training_loss += batch_loss.item()
            
            # Clear memory after each batch
            del inputs, query_points, targets, output, batch_loss
            if batch_idx % 5 == 0:  # More frequent cache clearing
                torch.cuda.empty_cache()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': training_loss / (batch_idx + 1)
            })
        
        # Calculate epoch metrics
        epoch_time = perf_counter() - epoch_start_time
        avg_training_loss = training_loss / len(train_loader)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_training_loss, epoch)
        writer.add_scalar('Time/Epoch', epoch_time, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_metrics_list = []
        total_val_loss = 0
        
        with torch.no_grad():
            for val_inputs, val_query_points, val_targets, _ in val_loader:
                val_inputs = val_inputs.to(device)
                val_query_points = val_query_points.to(device)
                val_targets = val_targets.to(device)
                
                val_output = model(val_inputs, val_query_points)
                val_loss = criterion(val_output, val_targets)
                
                # Calculate metrics for each sample in batch
                for j in range(val_inputs.size(0)):
                    metrics = calculate_metrics(val_output[j:j+1], val_targets[j:j+1])
                    val_metrics_list.append(metrics)
                    total_val_loss += val_loss.item()
            
            # Calculate average validation metrics
            avg_metrics = {
                'loss': total_val_loss / len(val_loader.dataset),
                'iou': np.mean([m['iou'] for m in val_metrics_list]),
                'precision': np.mean([m['precision'] for m in val_metrics_list]),
                'recall': np.mean([m['recall'] for m in val_metrics_list]),
                'f1_score': np.mean([m['f1_score'] for m in val_metrics_list])
            }
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/Validation', avg_metrics['loss'], epoch)
            writer.add_scalar('Metrics/IoU', avg_metrics['iou'], epoch)
            writer.add_scalar('Metrics/Precision', avg_metrics['precision'], epoch)
            writer.add_scalar('Metrics/Recall', avg_metrics['recall'], epoch)
            writer.add_scalar('Metrics/F1', avg_metrics['f1_score'], epoch)
            
            # Log epoch summary
            logging.info(
                f"\nEpoch {epoch} Summary:\n"
                f"Training Loss: {avg_training_loss:.4f}\n"
                f"Validation Loss: {avg_metrics['loss']:.4f}\n"
                f"Validation IoU: {avg_metrics['iou']:.4f}\n"
                f"Validation F1: {avg_metrics['f1_score']:.4f}"
            )
            
            # Visualization every 20 epochs
            if epoch % 20 == 0:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    
                    # Get a fresh batch of validation samples
                    viz_inputs, viz_queries, viz_skeletons, _ = next(iter(val_loader))
                    
                    # Select 2 samples for visualization
                    num_viz = min(2, len(viz_inputs))
                    
                    for i in range(num_viz):
                        viz_input = viz_inputs[i:i+1].to(device)
                        viz_query = viz_queries[i:i+1].to(device)
                        viz_skeleton = viz_skeletons[i:i+1].to(device)
                        
                        viz_output = model(viz_input, viz_query)
                        viz_mask = torch.zeros_like(viz_input)
                        viz_mask[0, 0, viz_query[0, 0], viz_query[0, 1]] = 1
                        
                        # Save visualization
                        fig = visualize_all_predictions(
                            viz_input.cpu(),
                            viz_mask.cpu(),
                            viz_output.cpu(),
                            viz_skeleton.cpu(),
                            f'visualizations/epoch_{epoch}_sample_{i}.png'
                        )
                        
                        # Log figure to TensorBoard
                        writer.add_figure(f'Predictions/Sample_{i}', fig, epoch)
                        plt.close(fig)
                        
                        del viz_output, viz_mask
                    
                    torch.cuda.empty_cache()
                    logging.info(f"Saved visualizations for epoch {epoch}")
            
            # Save periodically and when we find a new best model
            best_val_loss = float('inf')
            
            # Save best model when we find a better validation loss
            if avg_metrics['loss'] < best_val_loss:
                best_val_loss = avg_metrics['loss']
                logging.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'models/best_model.pt')
            
            # Also save periodic checkpoints
            if epoch % 50 == 0:  # Save every 50 epochs
                logging.info(f"Saving periodic checkpoint at epoch {epoch}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_metrics['loss'],
                }, f'models/model_epoch_{epoch}.pt')
        
        # Log gradient norm
        writer.add_scalar('Gradients/Average_Norm', avg_grad_norm, epoch)
    
    writer.close()
    logging.info(f"Training completed.")
    return {
        'epochs_trained': epoch + 1
    }
