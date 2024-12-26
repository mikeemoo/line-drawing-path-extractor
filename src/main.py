import logging
import wandb
import signal
import os
import sys
import traceback
import argparse
import matplotlib
import torch
import torch.nn as nn
matplotlib.use('Agg')

from .models.path_extraction_net import PathExtractionNet
from .losses.f1_loss import F1Loss
from .losses.combined_loss import CombinedLoss
from .utils.training import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Cleaning up...")
        if wandb.run is not None:
            try:
                wandb.finish()
                print("Wandb cleanup completed")
            except Exception as e:
                print(f"Error during wandb cleanup: {e}")
        print("Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Terminal kill

def get_default_sweep_config():
    """Define the default hyperparameter sweep configuration."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'training_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'hidden_channels': {
                'distribution': 'q_log_uniform_values',
                'q': 8,  # Round to nearest multiple of 8 for GPU efficiency
                'min': 16,
                'max': 64
            }
        }
    }
    return sweep_config

def default_config():
    config = {
        "learning_rate": {
            "value": 0.001
        },
        "gradient_clip_value": {
            "value": 0.9
        },
        "batch_size": {
            "value": 8
        },
        "hidden_channels": {
            "value": 48
        }
    }
    return config

def train_sweep(config=None):
    """Training function for wandb sweep."""
    try:
        with wandb.init(config=config):
            config = wandb.config
            
            model = PathExtractionNet(hidden_channels=config.hidden_channels)
            
            criterion = CombinedLoss(mse_scale=50.0, f1_weight=0.5)
            
            train_model(
                model=model,
                num_epochs=1000,
                batch_size=8, 
                learning_rate=getattr(config, 'learning_rate', 0.001),
                gradient_clip_value=getattr(config, 'gradient_clip_value', 0.9),
                criterion=criterion,
                hidden_channels=config.hidden_channels
            )
            
    except Exception as e:
        print(f"\nError in sweep run: {e}")
        traceback.print_exc()

def run_sweep(sweep_id=None):
    """Run or resume sweep with simple interrupt handling."""
    if sweep_id is None:
        sweep_config = get_default_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="path-extraction")
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Resuming sweep with ID: {sweep_id}")
    
    try:
        wandb.agent(
            sweep_id,
            train_sweep,
            count=20,
            project="path-extraction"
        )
    except Exception as e:
        print(f"\nError in sweep: {e}")
        traceback.print_exc()
    finally:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Error during wandb cleanup: {e}")
        print("\nExiting sweep...")
        print(f"To resume this sweep later, use sweep ID: {sweep_id}")

def main():
    setup_signal_handlers()
    
    try:
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"Using GPU: {device}")
            # Set memory allocation to be more efficient
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
                
        parser = argparse.ArgumentParser()
        parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
        parser.add_argument('--resume-sweep', type=str, help='Resume sweep with given ID')
        parser.add_argument('--resume', type=str, help='Resume from checkpoint')
        parser.add_argument('--wandb_mode', type=str, default='online', 
                          choices=['online', 'offline', 'disabled'],
                          help='wandb mode (online, offline, disabled)')
        args = parser.parse_args()
        
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['WANDB_MODE'] = args.wandb_mode
        
        if args.sweep or args.resume_sweep:
            run_sweep(sweep_id=args.resume_sweep)
        else:
            config = default_config()
            unwrapped_config = {k: v['value'] for k, v in config.items() if k != '_wandb'}
            
            if args.wandb_mode != 'disabled':
                wandb.init(
                    project="path-extraction",
                    config=unwrapped_config
                )
            
            model = PathExtractionNet(hidden_channels=unwrapped_config.get('hidden_channels', 64))
            
            criterion = CombinedLoss(mse_scale=50.0, f1_weight=0.5)
            
            if args.resume:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
            
            # Train with default hyperparameters
            train_model(
                model=model,
                num_epochs=100000,
                batch_size=8,
                learning_rate=unwrapped_config.get('learning_rate', 0.001),
                gradient_clip_value=unwrapped_config.get('gradient_clip_value', 0.9),
                criterion=criterion,
                hidden_channels=unwrapped_config.get('hidden_channels', 64)  # Pass hidden_channels
            )

    except Exception as e:
        print(f"\nError occurred: {e}")
        traceback.print_exc()
    finally:
        if wandb.run is not None:
            try:
                wandb.finish()
                print("Wandb cleanup completed")
            except Exception as e:
                print(f"Error during wandb cleanup: {e}")
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
