"""Data handling and preprocessing for plant disease detection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SyntheticPlantDataset(Dataset):
    """Synthetic plant disease dataset for demonstration purposes.
    
    This dataset generates synthetic leaf images to simulate healthy and diseased plants.
    In a real scenario, you would replace this with actual plant image data.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        """Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
            transform: Optional transforms to apply
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.seed = seed
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self) -> None:
        """Generate synthetic leaf images."""
        logger.info(f"Generating {self.num_samples} synthetic leaf images...")
        
        # Create base healthy leaf images (more uniform texture)
        healthy_images = []
        diseased_images = []
        
        for i in range(self.num_samples // 2):
            # Generate healthy leaf (uniform green texture)
            healthy_img = self._generate_healthy_leaf()
            healthy_images.append(healthy_img)
            
            # Generate diseased leaf (add spots/patches)
            diseased_img = self._generate_diseased_leaf(healthy_img)
            diseased_images.append(diseased_img)
        
        # Combine images and labels
        self.images = healthy_images + diseased_images
        self.labels = [0] * (self.num_samples // 2) + [1] * (self.num_samples // 2)
        
        logger.info(f"Generated {len(self.images)} images: {self.num_samples // 2} healthy, {self.num_samples // 2} diseased")
    
    def _generate_healthy_leaf(self) -> np.ndarray:
        """Generate a synthetic healthy leaf image.
        
        Returns:
            np.ndarray: Healthy leaf image as RGB array
        """
        # Create base green texture
        img = np.random.normal(loc=0.3, scale=0.05, size=(self.image_size, self.image_size, 3))
        
        # Add leaf-like structure (elliptical shape)
        center_x, center_y = self.image_size // 2, self.image_size // 2
        y, x = np.ogrid[:self.image_size, :self.image_size]
        
        # Create elliptical mask
        mask = ((x - center_x) ** 2 / (self.image_size * 0.4) ** 2 + 
                (y - center_y) ** 2 / (self.image_size * 0.6) ** 2) <= 1
        
        # Apply mask and adjust colors
        img[~mask] = 0
        img[mask] += np.random.normal(loc=0.1, scale=0.02, size=img[mask].shape)
        
        # Ensure RGB values are in [0, 1] range
        img = np.clip(img, 0, 1)
        
        return img
    
    def _generate_diseased_leaf(self, healthy_img: np.ndarray) -> np.ndarray:
        """Generate a synthetic diseased leaf image.
        
        Args:
            healthy_img: Base healthy leaf image
            
        Returns:
            np.ndarray: Diseased leaf image as RGB array
        """
        diseased_img = healthy_img.copy()
        
        # Add disease spots (dark patches)
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            # Random spot position and size
            spot_x = np.random.randint(0, self.image_size)
            spot_y = np.random.randint(0, self.image_size)
            spot_size = np.random.randint(5, 15)
            
            # Create circular spot
            y, x = np.ogrid[:self.image_size, :self.image_size]
            spot_mask = ((x - spot_x) ** 2 + (y - spot_y) ** 2) <= spot_size ** 2
            
            # Add dark brown/black spots
            diseased_img[spot_mask] *= 0.3
            diseased_img[spot_mask] += np.random.normal(loc=0.1, scale=0.05, size=diseased_img[spot_mask].shape)
        
        # Add some yellowing (chlorosis)
        yellowing_mask = np.random.random((self.image_size, self.image_size)) < 0.1
        diseased_img[yellowing_mask, 0] += 0.2  # Increase red
        diseased_img[yellowing_mask, 1] += 0.1  # Increase green
        
        # Ensure RGB values are in [0, 1] range
        diseased_img = np.clip(diseased_img, 0, 1)
        
        return diseased_img
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(config: DictConfig, is_training: bool = True) -> transforms.Compose:
    """Get image transforms based on configuration.
    
    Args:
        config: Configuration object
        is_training: Whether transforms are for training
        
    Returns:
        transforms.Compose: Composed transforms
    """
    transform_list = [
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    if is_training and config.data.augmentations:
        # Add augmentation transforms
        aug_config = config.data.augmentations
        
        augmentation_transforms = []
        
        if aug_config.get("horizontal_flip", 0) > 0:
            augmentation_transforms.append(
                transforms.RandomHorizontalFlip(p=aug_config.horizontal_flip)
            )
        
        if aug_config.get("rotation", 0) > 0:
            augmentation_transforms.append(
                transforms.RandomRotation(degrees=aug_config.rotation)
            )
        
        if any(aug_config.get(key, 0) > 0 for key in ["brightness", "contrast", "saturation"]):
            augmentation_transforms.append(
                transforms.ColorJitter(
                    brightness=aug_config.get("brightness", 0),
                    contrast=aug_config.get("contrast", 0),
                    saturation=aug_config.get("saturation", 0),
                )
            )
        
        # Insert augmentation transforms before normalization
        transform_list = transform_list[:1] + augmentation_transforms + transform_list[1:]
    
    return transforms.Compose(transform_list)


def create_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SyntheticPlantDataset(
        num_samples=int(config.data.train_split * 1000),
        image_size=config.data.image_size,
        transform=get_transforms(config, is_training=True),
        seed=42,
    )
    
    val_dataset = SyntheticPlantDataset(
        num_samples=int(config.data.val_split * 1000),
        image_size=config.data.image_size,
        transform=get_transforms(config, is_training=False),
        seed=43,
    )
    
    test_dataset = SyntheticPlantDataset(
        num_samples=int(config.data.test_split * 1000),
        image_size=config.data.image_size,
        transform=get_transforms(config, is_training=False),
        seed=44,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader
