"""Model architectures for plant disease detection."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """Custom CNN architecture for plant disease detection."""
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        input_channels: int = 3,
    ):
        """Initialize custom CNN.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
            input_channels: Number of input channels
        """
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PlantDiseaseModel(nn.Module):
    """Plant disease detection model wrapper."""
    
    def __init__(self, config: DictConfig):
        """Initialize model.
        
        Args:
            config: Configuration object
        """
        super(PlantDiseaseModel, self).__init__()
        
        self.config = config
        self.num_classes = config.model.num_classes
        
        # Create backbone model
        if config.model.architecture == "custom_cnn":
            self.backbone = CustomCNN(
                num_classes=self.num_classes,
                dropout=config.model.dropout,
            )
        elif config.model.architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=config.model.pretrained)
            # Modify final layer
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        elif config.model.architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=config.model.pretrained)
            # Modify final layer
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        elif config.model.architecture == "efficientnet_b0":
            try:
                import torchvision.models as models
                self.backbone = models.efficientnet_b0(pretrained=config.model.pretrained)
                # Modify final layer
                self.backbone.classifier[1] = nn.Linear(
                    self.backbone.classifier[1].in_features, 
                    self.num_classes
                )
            except ImportError:
                logger.warning("EfficientNet not available, falling back to ResNet18")
                self.backbone = models.resnet18(pretrained=config.model.pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {config.model.architecture}")
        
        # Add dropout if specified
        if config.model.dropout > 0 and config.model.architecture != "custom_cnn":
            if hasattr(self.backbone, 'fc'):
                # For ResNet
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(config.model.dropout),
                    self.backbone.fc
                )
            elif hasattr(self.backbone, 'classifier'):
                # For EfficientNet
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(config.model.dropout),
                    self.backbone.classifier
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.backbone(x)
    
    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (backbone without final classification layer).
        
        Returns:
            Feature extractor module
        """
        if hasattr(self.backbone, 'fc'):
            # For ResNet
            feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        elif hasattr(self.backbone, 'classifier'):
            # For EfficientNet
            feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            # For custom CNN
            feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        return feature_extractor


def create_model(config: DictConfig) -> PlantDiseaseModel:
    """Create model based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        PlantDiseaseModel: Initialized model
    """
    model = PlantDiseaseModel(config)
    
    logger.info(f"Created model: {config.model.architecture}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def create_optimizer(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration object
        
    Returns:
        Optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    
    logger.info(f"Created Adam optimizer with lr={config.model.learning_rate}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: DictConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Configuration object
        
    Returns:
        Learning rate scheduler or None
    """
    # Simple step scheduler - can be extended based on needs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.epochs // 3,
        gamma=0.1,
    )
    
    logger.info("Created StepLR scheduler")
    
    return scheduler
