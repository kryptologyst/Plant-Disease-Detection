"""Unit tests for plant disease detection models."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models import create_model, PlantDiseaseModel, CustomCNN
from src.utils import get_device, set_seed, count_parameters


class TestCustomCNN:
    """Test cases for CustomCNN model."""
    
    def test_custom_cnn_initialization(self):
        """Test CustomCNN initialization."""
        model = CustomCNN(num_classes=2, dropout=0.5)
        assert isinstance(model, CustomCNN)
        assert model.classifier[-1].out_features == 2
    
    def test_custom_cnn_forward(self):
        """Test CustomCNN forward pass."""
        model = CustomCNN(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_custom_cnn_parameters(self):
        """Test CustomCNN parameter count."""
        model = CustomCNN(num_classes=2)
        param_count = count_parameters(model)
        assert param_count > 0
        assert param_count < 1000000  # Reasonable upper bound


class TestPlantDiseaseModel:
    """Test cases for PlantDiseaseModel wrapper."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config_dict = {
            'model': {
                'architecture': 'resnet18',
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.5,
                'learning_rate': 0.001,
                'weight_decay': 1e-4
            },
            'training': {
                'epochs': 10,
                'early_stopping_patience': 5,
                'save_best_only': True,
                'monitor': 'val_accuracy',
                'mode': 'max'
            },
            'paths': {
                'model_dir': 'models',
                'assets_dir': 'assets'
            }
        }
        return OmegaConf.create(config_dict)
    
    def test_plant_disease_model_initialization(self, config):
        """Test PlantDiseaseModel initialization."""
        model = PlantDiseaseModel(config)
        assert isinstance(model, PlantDiseaseModel)
        assert model.num_classes == 2
    
    def test_plant_disease_model_forward(self, config):
        """Test PlantDiseaseModel forward pass."""
        model = PlantDiseaseModel(config)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_create_model_function(self, config):
        """Test create_model function."""
        model = create_model(config)
        assert isinstance(model, PlantDiseaseModel)
        assert count_parameters(model) > 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are reproducible
        torch.manual_seed(42)
        rand1 = torch.randn(1)
        torch.manual_seed(42)
        rand2 = torch.randn(1)
        assert torch.allclose(rand1, rand2)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = CustomCNN(num_classes=2)
        param_count = count_parameters(model)
        assert isinstance(param_count, int)
        assert param_count > 0


class TestModelArchitectures:
    """Test different model architectures."""
    
    @pytest.fixture
    def architectures(self):
        """List of architectures to test."""
        return ['custom_cnn', 'resnet18']
    
    def test_all_architectures(self, architectures):
        """Test all supported architectures."""
        for arch in architectures:
            config_dict = {
                'model': {
                    'architecture': arch,
                    'num_classes': 2,
                    'pretrained': False,
                    'dropout': 0.5,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-4
                },
                'training': {
                    'epochs': 10,
                    'early_stopping_patience': 5,
                    'save_best_only': True,
                    'monitor': 'val_accuracy',
                    'mode': 'max'
                },
                'paths': {
                    'model_dir': 'models',
                    'assets_dir': 'assets'
                }
            }
            config = OmegaConf.create(config_dict)
            
            model = create_model(config)
            assert isinstance(model, PlantDiseaseModel)
            
            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
