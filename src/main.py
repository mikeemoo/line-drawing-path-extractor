import logging
import wandb
import signal
import os
import sys
import traceback
import argparse
import matplotlib
import torch
matplotlib.use('Agg')

from .models.skeletonization import Skeletonization
from .models.skeleton_repair_net import SkeletonRepairNet
from .losses.f1_loss import F1Loss
from .losses.repair_loss import SkeletonRepairLoss
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
            'name': 'epoch_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'gradient_clip_value': {
                'distribution': 'log_uniform_values',
                'min': 0.85,
                'max': 0.95
            }
        }
    }
    return sweep_config

def default_config():
    config = {
        "decoder_kernel_size": {
            "value": 5
        },
        "decoder_residual_blocks": {
            "value": 1
        }
    }
    return config

def train_sweep(config=None):
    """Training function for wandb sweep."""
    try:
        with wandb.init(config=config):
            config = wandb.config
            
            # Initialize main model with its parameters
            model = Skeletonization(
                dropout_rate=getattr(config, 'dropout_rate', 0.35),
                base_channels=getattr(config, 'base_channels', 64),
                decoder_kernel_size=getattr(config, 'decoder_kernel_size', 3),
                decoder_residual_blocks=getattr(config, 'decoder_residual_blocks', 1)
            )
            
            # Initialize repair model with correct parameters
            repair_model = SkeletonRepairNet(in_channels=1)
            
            criterion = F1Loss()
            repair_criterion = SkeletonRepairLoss()
            
            train_model(
                model=model,
                repair_model=repair_model,
                num_epochs=1000,
                batch_size=getattr(config, 'batch_size', 32),
                learning_rate=getattr(config, 'learning_rate', 0.008),
                gradient_clip_value=getattr(config, 'gradient_clip_value', 0.6),
                criterion=criterion,
                repair_criterion=repair_criterion
            )
            
    except Exception as e:
        print(f"\nError in sweep run: {e}")
        traceback.print_exc()

def run_sweep(sweep_id=None):
    """Run or resume sweep with simple interrupt handling."""
    if sweep_id is None:
        sweep_config = get_default_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="fluxnet-training")
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Resuming sweep with ID: {sweep_id}")
    
    try:
        wandb.agent(
            sweep_id,
            train_sweep,
            count=1,
            project="fluxnet-training"
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
                
        parser = argparse.ArgumentParser()
        parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
        parser.add_argument('--resume-sweep', type=str, help='Resume sweep with given ID')
        parser.add_argument('--resume', type=str, help='Resume from checkpoint')
        parser.add_argument('--wandb_mode', type=str, default='online', 
                          choices=['online', 'offline', 'disabled'],
                          help='wandb mode (online, offline, disabled)')
        args = parser.parse_args()
        
        # Set wandb mode
        os.environ['WANDB_MODE'] = args.wandb_mode
        
        if args.sweep or args.resume_sweep:
            run_sweep(sweep_id=args.resume_sweep)
        else:
            # Use default configuration
            config = default_config()
            unwrapped_config = {k: v['value'] for k, v in config.items() if k != '_wandb'}
            
            # Initialize wandb with default config
            if args.wandb_mode != 'disabled':
                wandb.init(
                    project="fluxnet-training",
                    config=unwrapped_config
                )
            
            # Initialize main model with its parameters
            model = Skeletonization(
                dropout_rate=getattr(unwrapped_config, 'dropout_rate', 0.35),
                base_channels=getattr(unwrapped_config, 'base_channels', 64),
                decoder_kernel_size=getattr(unwrapped_config, 'decoder_kernel_size', 3),
                decoder_residual_blocks=getattr(unwrapped_config, 'decoder_residual_blocks', 1)
            )
            
            # Initialize repair model with correct parameters
            repair_model = SkeletonRepairNet(in_channels=1)
            
            criterion = F1Loss()
            repair_criterion = SkeletonRepairLoss()
            
            if args.resume:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state'])
                repair_model.load_state_dict(checkpoint['repair_model_state'])
                print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
            
            # Train with default hyperparameters
            train_model(
                model=model,
                repair_model=repair_model,
                num_epochs=150,
                batch_size=getattr(unwrapped_config, 'batch_size', 32),
                learning_rate=getattr(unwrapped_config, 'learning_rate', 0.0015),
                gradient_clip_value=getattr(unwrapped_config, 'gradient_clip_value', 0.6),
                criterion=criterion,
                repair_criterion=repair_criterion
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
