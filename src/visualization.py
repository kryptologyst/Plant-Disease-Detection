"""Visualization utilities for plant disease detection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training and validation accuracy
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning curves comparison
    axes[1, 0].plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    axes[1, 0].plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    axes[1, 0].set_title('Loss Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot accuracy comparison
    axes[1, 1].plot(history['train_acc'], label='Training Accuracy', color='blue', alpha=0.7)
    axes[1, 1].plot(history['val_acc'], label='Validation Accuracy', color='red', alpha=0.7)
    axes[1, 1].set_title('Accuracy Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = ['Healthy', 'Diseased']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 8,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot sample predictions with ground truth.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to plot
        class_names: Optional class names
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = ['Healthy', 'Diseased']
    
    model.eval()
    
    # Get a batch of samples
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Take only the requested number of samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Move to device and get predictions
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
    
    # Move back to CPU for plotting
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # Create subplots
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    if num_samples == 1:
        axes = [axes]
    elif num_samples <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('Sample Predictions', fontsize=16)
    
    for i in range(num_samples):
        # Denormalize image for display
        img = images[i]
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        
        # Plot image
        axes[i].imshow(img)
        
        # Get prediction info
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]].item()
        
        # Set title with prediction info
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
            color=color
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison of metrics across different models.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Optional path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(metrics_dict)
    
    for i, (model_name, model_metrics) in enumerate(metrics_dict.items()):
        values = [model_metrics.get(metric, 0) for metric in metrics]
        ax.bar(x + i * width, values, width, label=model_name)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()


def create_evaluation_report(
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
    save_dir: str,
) -> None:
    """Create comprehensive evaluation report.
    
    Args:
        metrics: Evaluation metrics
        history: Training history
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    plot_training_history(history, save_path=str(save_dir / "training_history.png"))
    
    # Create metrics summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Evaluation report saved to {save_dir}")
