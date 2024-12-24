import torch
import logging
import wandb
import os
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def train_model(model, repair_model, num_epochs=50, batch_size=32, learning_rate=0.004, 
                gradient_clip_value=1.0, criterion=None, repair_criterion=None):
    """Training loop with configurable skeleton loss and repair model.
    
    The training process handles two models in parallel:
    1. Main Skeletonization model: input image [B, 1, H, W] -> skeleton [B, 1, H, W]
    2. Repair model: damaged skeleton [B, 1, H, W] -> repaired skeleton [B, 1, H, W]
    
    where B = batch_size (default: 32), H, W = image dimensions (default: 64, 64)
    
    Args:
        model (Skeletonization): Main skeletonization model
        repair_model (SkeletonRepairNet): Model for repairing damaged skeletons
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        gradient_clip_value (float): Maximum gradient norm
        criterion: Loss function for main model
        repair_criterion: Loss function for repair model
    """
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Move models to device
    model = model.to(device)
    repair_model = repair_model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    repair_optimizer = torch.optim.AdamW(repair_model.parameters(), lr=learning_rate)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
    repair_scheduler = CosineAnnealingLR(repair_optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
    
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    wandb_enabled = wandb.run is not None
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        train_size=batch_size * 35,
        val_size=60,
        num_workers=4
    )
    
    # Get a fixed test image from the validation set
    viz_input, viz_skeleton, viz_damaged = next(iter(val_loader))
    viz_input = viz_input[0:1].to(device)
    viz_skeleton = viz_skeleton[0:1].to(device)
    viz_damaged = viz_damaged[0:1].to(device)
    
    for epoch in range(num_epochs):
        model.train()
        repair_model.train()
        
        training_loss = 0
        repair_training_loss = 0
        epoch_start_time = perf_counter()
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets, damaged) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            damaged = damaged.to(device)
            
            # Train main skeletonization model
            optimizer.zero_grad()
            output = model(inputs)
            batch_loss = criterion(output, targets)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            training_loss += batch_loss.item()
            
            # Train repair model
            repair_optimizer.zero_grad()
            repair_output = repair_model(damaged)
            repair_loss = repair_criterion(repair_output, targets)
            repair_loss.backward()
            torch.nn.utils.clip_grad_norm_(repair_model.parameters(), gradient_clip_value)
            repair_optimizer.step()
            repair_training_loss += repair_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': batch_loss.item(),
                'repair_loss': repair_loss.item()
            })
        
        # Calculate epoch metrics
        epoch_time = perf_counter() - epoch_start_time
        avg_training_loss = training_loss / len(train_loader)
        avg_repair_loss = repair_training_loss / len(train_loader)
        
        # Log metrics
        log_dict = {
            "epoch": epoch,
            "training_loss": avg_training_loss,
            "repair_loss": avg_repair_loss,
            "epoch_time": epoch_time,
            "learning_rate": scheduler.get_last_lr()[0],
            "repair_learning_rate": repair_scheduler.get_last_lr()[0]
        }
        
        if wandb_enabled:
            wandb.log(log_dict)
        
        scheduler.step()
        repair_scheduler.step()
        
        # Validation phase
        model.eval()
        repair_model.eval()
        
        val_metrics_list = []
        repair_val_metrics_list = []
        total_val_loss = 0
        total_repair_val_loss = 0
        
        with torch.no_grad():
            for val_inputs, val_targets, val_damaged in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_damaged = val_damaged.to(device)
                
                # Validate main model
                val_output = model(val_inputs)
                val_loss = criterion(val_output, val_targets)
                
                # Validate repair model
                repair_output = repair_model(val_damaged)
                repair_val_loss = repair_criterion(repair_output, val_targets)
                
                # Calculate metrics for each sample in batch
                for j in range(val_inputs.size(0)):
                    metrics = calculate_metrics(val_output[j:j+1], val_targets[j:j+1])
                    repair_metrics = calculate_metrics(repair_output[j:j+1], val_targets[j:j+1])
                    
                    val_metrics_list.append(metrics)
                    repair_val_metrics_list.append(repair_metrics)
                    
                    total_val_loss += val_loss.item()
                    total_repair_val_loss += repair_val_loss.item()
            
            # Calculate average validation metrics
            avg_metrics = {
                'loss': total_val_loss / len(val_loader.dataset),
                'iou': np.mean([m['iou'] for m in val_metrics_list]),
                'precision': np.mean([m['precision'] for m in val_metrics_list]),
                'recall': np.mean([m['recall'] for m in val_metrics_list]),
                'f1_score': np.mean([m['f1_score'] for m in val_metrics_list])
            }
            
            avg_repair_metrics = {
                'loss': total_repair_val_loss / len(val_loader.dataset),
                'iou': np.mean([m['iou'] for m in repair_val_metrics_list]),
                'precision': np.mean([m['precision'] for m in repair_val_metrics_list]),
                'recall': np.mean([m['recall'] for m in repair_val_metrics_list]),
                'f1_score': np.mean([m['f1_score'] for m in repair_val_metrics_list])
            }
            
            # Log validation metrics
            if wandb_enabled:
                wandb.log({
                    'epoch': epoch,
                    'val_loss': avg_metrics['loss'],
                    'val_iou': avg_metrics['iou'],
                    'val_precision': avg_metrics['precision'],
                    'val_recall': avg_metrics['recall'],
                    'val_f1_score': avg_metrics['f1_score'],
                    'repair_val_loss': avg_repair_metrics['loss'],
                    'repair_val_iou': avg_repair_metrics['iou'],
                    'repair_val_precision': avg_repair_metrics['precision'],
                    'repair_val_recall': avg_repair_metrics['recall'],
                    'repair_val_f1_score': avg_repair_metrics['f1_score']
                })
            
            # Log to console
            logging.info(
                f"Validation: IoU: {avg_metrics['iou']:.4f}, "
                f"F1: {avg_metrics['f1_score']:.4f}, "
                f"Loss: {avg_metrics['loss']:.4f}"
            )
            logging.info(
                f"Repair Validation: IoU: {avg_repair_metrics['iou']:.4f}, "
                f"F1: {avg_repair_metrics['f1_score']:.4f}, "
                f"Loss: {avg_repair_metrics['loss']:.4f}"
            )
            
            # Early stopping check (based on main model loss)
            early_stopping(avg_training_loss)
            if early_stopping.early_stop:
                logging.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best loss: {early_stopping.best_loss:.4f}"
                )
                # Save models before breaking
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': early_stopping.best_loss,
                }, 'models/skeletonization_model.pth')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': repair_model.state_dict(),
                    'optimizer_state_dict': repair_optimizer.state_dict(),
                    'loss': avg_repair_metrics['loss'],
                }, 'models/repair_model.pth')
                
                break
            
            # Visualization using fixed test image
            if epoch % 10 == 0:
                viz_output = model(viz_input)
                repair_output = repair_model(viz_damaged)
                
                fig = visualize_all_predictions(
                    viz_input.squeeze().cpu().numpy(),
                    viz_skeleton.squeeze().cpu().numpy(),
                    viz_output.squeeze().cpu().numpy()
                )
                
                repair_fig = visualize_all_predictions(
                    viz_damaged.squeeze().cpu().numpy(),
                    viz_skeleton.squeeze().cpu().numpy(),
                    repair_output.squeeze().cpu().numpy()
                )
                
                # Save figures locally
                fig.savefig(f'visualizations/skeleton_prediction_epoch_{epoch}.png')
                repair_fig.savefig(f'visualizations/repair_prediction_epoch_{epoch}.png')
                
                if wandb_enabled:
                    wandb.log({
                        "test_image_progress": wandb.Image(fig),
                        "repair_test_image_progress": wandb.Image(repair_fig)
                    })
                plt.close(fig)
                plt.close(repair_fig) 

    # After training loop, save final models if not stopped early
    if not early_stopping.early_stop:
        torch.save({
            'epoch': num_epochs-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_training_loss,
        }, 'models/skeletonization_model.pth')
        
        torch.save({
            'epoch': num_epochs-1,
            'model_state_dict': repair_model.state_dict(),
            'optimizer_state_dict': repair_optimizer.state_dict(),
            'loss': avg_repair_metrics['loss'],
        }, 'models/repair_model.pth')
        
        logging.info("Training completed. Models saved in 'models' directory.") 
