import os
import logging
import argparse
import json
import torch
import torch.nn as nn

from .models.path_extraction_net import PathExtractionNet
from .utils.training import train_model
from .losses.combined_loss import CombinedLoss

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def cleanup():
    """Cleanup resources."""
    try:
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared")
    except Exception as e:
        logging.error(f"Error during CUDA cleanup: {e}")

def train_with_config(config):
    """Training function with configuration."""
    # Create model
    model = PathExtractionNet(
        in_channels=2,
        hidden_channels=config.get('hidden_channels', 64)
    )
    
    # Create loss function
    criterion = CombinedLoss(
        mse_scale=config.get('mse_scale', 50.0),
        f1_weight=config.get('f1_weight', 0.5),
        epsilon=config.get('epsilon', 1e-7)
    )
    
    # Train model
    train_model(
        model=model,
        num_epochs=config.get('num_epochs', 3000),
        batch_size=config.get('batch_size', 8),
        learning_rate=config.get('learning_rate', 0.001),
        gradient_clip_value=config.get('gradient_clip_value', 0.9),
        criterion=criterion,
        hidden_channels=config.get('hidden_channels', 48)
    )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train path extraction model')
    parser.add_argument('--config', type=str, default='config.json',
                      help='path to config file')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {args.config} not found, using defaults")
        config = {}
    
    try:
        # Train model
        train_with_config(config)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e
    finally:
        cleanup()

if __name__ == '__main__':
    main()
