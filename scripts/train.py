#!/usr/bin/env python3
"""Main training script for plant disease detection."""

import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import setup_logging, set_seed, get_device, create_directories
from src.data import create_data_loaders
from src.models import create_model
from src.training import train_model, evaluate_model
from src.visualization import plot_training_history, create_evaluation_report


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    set_seed(42)
    device = get_device()
    create_directories(config)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Train model
    logger.info("Starting training...")
    history = train_model(model, train_loader, val_loader, config, device)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_training_history(history, save_path=str(Path(config.paths.assets_dir) / "training_history.png"))
    create_evaluation_report(metrics, history, config.paths.assets_dir)
    
    # Save final model
    model_path = Path(config.paths.model_dir) / "final_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Final model saved to {model_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    import torch
    main()
