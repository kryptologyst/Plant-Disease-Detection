"""Streamlit demo app for plant disease detection."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .diseased {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class PlantDiseaseDetector:
    """Plant disease detection model wrapper for demo."""
    
    def __init__(self):
        """Initialize the detector."""
        self.device = self._get_device()
        self.model = None
        self.transform = self._get_transform()
        self.class_names = ['Healthy', 'Diseased']
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            if not Path(model_path).exists():
                return False
            
            # Create a simple model architecture
            import torchvision.models as models
            self.model = models.resnet18(pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, image: Image.Image) -> Tuple[str, float, np.ndarray]:
        """Predict plant disease from image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (prediction, confidence, probabilities)
        """
        if self.model is None:
            return "Model not loaded", 0.0, np.array([0.5, 0.5])
        
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = outputs.argmax(dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            pred_class = self.class_names[prediction]
            probs = probabilities[0].cpu().numpy()
            
            return pred_class, confidence, probs
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "Error", 0.0, np.array([0.5, 0.5])


def generate_synthetic_image(is_healthy: bool = True) -> Image.Image:
    """Generate a synthetic plant leaf image for demo purposes.
    
    Args:
        is_healthy: Whether to generate a healthy or diseased leaf
        
    Returns:
        PIL Image
    """
    size = 224
    
    if is_healthy:
        # Generate healthy leaf
        img = np.random.normal(loc=0.3, scale=0.05, size=(size, size, 3))
        
        # Add leaf-like structure
        center_x, center_y = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        mask = ((x - center_x) ** 2 / (size * 0.4) ** 2 + 
                (y - center_y) ** 2 / (size * 0.6) ** 2) <= 1
        
        img[~mask] = 0
        img[mask] += np.random.normal(loc=0.1, scale=0.02, size=np.sum(mask))
    else:
        # Generate diseased leaf
        img = np.random.normal(loc=0.3, scale=0.05, size=(size, size, 3))
        
        # Add leaf-like structure
        center_x, center_y = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        mask = ((x - center_x) ** 2 / (size * 0.4) ** 2 + 
                (y - center_y) ** 2 / (size * 0.6) ** 2) <= 1
        
        img[~mask] = 0
        img[mask] += np.random.normal(loc=0.1, scale=0.02, size=np.sum(mask))
        
        # Add disease spots
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            spot_x = np.random.randint(0, size)
            spot_y = np.random.randint(0, size)
            spot_size = np.random.randint(5, 15)
            
            spot_mask = ((x - spot_x) ** 2 + (y - spot_y) ** 2) <= spot_size ** 2
            img[spot_mask] *= 0.3
            img[spot_mask] += np.random.normal(loc=0.1, scale=0.05, size=(3,))
    
    # Ensure RGB values are in [0, 1] range
    img = np.clip(img, 0, 1)
    
    return Image.fromarray((img * 255).astype(np.uint8))


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection</h1>', unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = PlantDiseaseDetector()
    
    detector = st.session_state.detector
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model")
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="models/best_model.pth",
        help="Path to the trained model file"
    )
    
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            if detector.load_model(model_path):
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model. Using synthetic predictions.")
    
    # Image input
    st.sidebar.subheader("Image Input")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Generate Synthetic", "Use Sample"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Image")
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a plant leaf image for disease detection"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                st.info("Please upload an image file")
                image = None
        
        elif input_method == "Generate Synthetic":
            col_gen1, col_gen2 = st.columns(2)
            
            with col_gen1:
                if st.button("Generate Healthy Leaf"):
                    image = generate_synthetic_image(is_healthy=True)
                    st.session_state.generated_image = image
                    st.session_state.generated_label = "Healthy"
            
            with col_gen2:
                if st.button("Generate Diseased Leaf"):
                    image = generate_synthetic_image(is_healthy=False)
                    st.session_state.generated_image = image
                    st.session_state.generated_label = "Diseased"
            
            if 'generated_image' in st.session_state:
                image = st.session_state.generated_image
                label = st.session_state.generated_label
                st.image(image, caption=f"Generated {label} Leaf", use_column_width=True)
            else:
                image = None
        
        else:  # Use Sample
            sample_type = st.selectbox("Sample Type", ["Healthy", "Diseased"])
            if st.button("Load Sample"):
                image = generate_synthetic_image(is_healthy=(sample_type == "Healthy"))
                st.image(image, caption=f"Sample {sample_type} Leaf", use_column_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if image is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                prediction, confidence, probabilities = detector.predict(image)
            
            # Display results
            if prediction == "Healthy":
                st.markdown(
                    f'<div class="prediction-result healthy">'
                    f'✅ Prediction: {prediction}<br>'
                    f'Confidence: {confidence:.2%}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif prediction == "Diseased":
                st.markdown(
                    f'<div class="prediction-result diseased">'
                    f'⚠️ Prediction: {prediction}<br>'
                    f'Confidence: {confidence:.2%}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error(f"Error: {prediction}")
            
            # Probability distribution
            st.subheader("Probability Distribution")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(detector.class_names, probabilities, 
                         color=['#28a745', '#dc3545'], alpha=0.7)
            ax.set_title('Prediction Probabilities')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Metrics
            st.subheader("Prediction Metrics")
            
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col_met2:
                uncertainty = 1 - confidence
                st.metric("Uncertainty", f"{uncertainty:.2%}")
        
        else:
            st.info("Please provide an image to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Plant Disease Detection Demo | Author: <a href='https://github.com/kryptologyst'>kryptologyst</a></p>
        <p><strong>Disclaimer:</strong> This is a research demonstration. Not intended for operational use.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
