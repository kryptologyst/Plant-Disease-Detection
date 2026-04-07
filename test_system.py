#!/usr/bin/env python3
"""Quick test script to verify the plant disease detection system."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import set_seed, get_device, create_directories
        from src.data import create_data_loaders, SyntheticPlantDataset
        from src.models import create_model
        from src.training import train_model, evaluate_model
        from src.visualization import plot_training_history
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.utils import set_seed, get_device
        from omegaconf import OmegaConf
        
        # Test seed setting
        set_seed(42)
        print("✅ Seed setting works")
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection works: {device}")
        
        # Test configuration
        config_dict = {
            'data': {
                'image_size': 64,
                'batch_size': 8,
                'num_workers': 2,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'augmentations': {
                    'horizontal_flip': 0.5,
                    'rotation': 10,
                    'brightness': 0.1,
                    'contrast': 0.1,
                    'saturation': 0.1
                }
            },
            'model': {
                'architecture': 'custom_cnn',
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.3,
                'learning_rate': 0.01,
                'weight_decay': 1e-4
            },
            'training': {
                'epochs': 2,
                'early_stopping_patience': 1,
                'save_best_only': True,
                'monitor': 'val_accuracy',
                'mode': 'max'
            },
            'paths': {
                'data_dir': 'data',
                'raw_data_dir': 'data/raw',
                'processed_data_dir': 'data/processed',
                'model_dir': 'models',
                'output_dir': 'outputs',
                'assets_dir': 'assets'
            }
        }
        
        config = OmegaConf.create(config_dict)
        print("✅ Configuration creation works")
        
        # Test data loading
        from src.data import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        print(f"✅ Data loaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
        # Test model creation
        from src.models import create_model
        model = create_model(config)
        print(f"✅ Model created: {config.model.architecture}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_demo_app():
    """Test demo app syntax."""
    print("\nTesting demo app...")
    
    try:
        import py_compile
        py_compile.compile('demo/app.py', doraise=True)
        print("✅ Demo app syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Demo app syntax error: {e}")
        return False

def main():
    """Run all tests."""
    print("🌱 Plant Disease Detection - System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_demo_app,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Launch demo: streamlit run demo/app.py")
        print("3. Check out the notebook: jupyter notebook notebooks/demo.ipynb")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
