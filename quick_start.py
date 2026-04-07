#!/usr/bin/env python3
"""Quick start script for plant disease detection."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main quick start function."""
    print("🌱 Plant Disease Detection - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Test system
    print("\n1. Testing system...")
    if not run_command("python3 test_system.py", "System test"):
        print("❌ System test failed. Please check your installation.")
        return 1
    
    # Ask user what they want to do
    print("\n" + "=" * 50)
    print("What would you like to do next?")
    print("1. Run a quick training demo (2 epochs)")
    print("2. Launch the Streamlit demo app")
    print("3. Open Jupyter notebook")
    print("4. Run full training")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Running quick training demo...")
        cmd = """
        python3 -c "
        import sys
        sys.path.append('.')
        from src.utils import set_seed, get_device, create_directories
        from src.data import create_data_loaders
        from src.models import create_model
        from src.training import train_model, evaluate_model
        from omegaconf import OmegaConf
        
        # Quick config
        config_dict = {
            'data': {
                'image_size': 128,
                'batch_size': 16,
                'num_workers': 2,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'augmentations': {
                    'horizontal_flip': 0.5,
                    'rotation': 15,
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2
                }
            },
            'model': {
                'architecture': 'custom_cnn',
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.3,
                'learning_rate': 0.001,
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
        
        # Setup
        set_seed(42)
        device = get_device()
        create_directories(config)
        
        print(f'Using device: {device}')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        print(f'Data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}')
        
        # Create model
        model = create_model(config)
        model = model.to(device)
        print(f'Model: {config.model.architecture}')
        
        # Train
        history = train_model(model, train_loader, val_loader, config, device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        print('\\nFinal Results:')
        for metric, value in metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}')
        
        print('\\n🎉 Quick demo completed!')
        "
        """
        run_command(cmd, "Quick training demo")
        
    elif choice == "2":
        print("\n🚀 Launching Streamlit demo...")
        print("The demo will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the demo")
        try:
            subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        except FileNotFoundError:
            print("❌ Streamlit not found. Please install it with: pip install streamlit")
            
    elif choice == "3":
        print("\n🚀 Opening Jupyter notebook...")
        print("The notebook will open in your browser")
        print("Press Ctrl+C to stop Jupyter")
        try:
            subprocess.run(["jupyter", "notebook", "notebooks/demo.ipynb"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Jupyter stopped by user")
        except FileNotFoundError:
            print("❌ Jupyter not found. Please install it with: pip install jupyter")
            
    elif choice == "4":
        print("\n🚀 Running full training...")
        run_command("python3 scripts/train.py", "Full training")
        
    elif choice == "5":
        print("\n👋 Goodbye!")
        return 0
        
    else:
        print("❌ Invalid choice. Please run the script again.")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 Thanks for using Plant Disease Detection!")
    print("📚 Check the README.md for more information")
    print("🐛 Report issues at: https://github.com/kryptologyst")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
