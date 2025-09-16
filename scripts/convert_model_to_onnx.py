#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import models
import os
import json

# EXACT COPY of BrainTumorCNN class from training
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Match your exact unfreezing pattern
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
            
        # EXACT MATCH of enhanced classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def convert_your_model_to_onnx():
    """
    Convert your exact trained model to ONNX format for web deployment
    """
    try:
        from config import Config
        Config.create_directories()
        model_path = Config.BEST_MODEL_PATH
        output_dir = "brain-tumor-classifier/public/model"
        
        # Get training data to determine classes
        data_dir = Config.get_data_path()
        if data_dir:
            from torchvision import datasets
            dummy_dataset = datasets.ImageFolder(root=data_dir)
            class_names = dummy_dataset.classes
            print(f"Detected classes from training data: {class_names}")
        else:
            # Fallback to common brain tumor classes
            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            print(f"Using fallback classes: {class_names}")
            
    except ImportError:
        print("Config not found, using fallback paths...")
        model_path = "models/best_brain_tumor_model.pth"
        output_dir = "brain-tumor-classifier/public/model"
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train your model first by running: python train.py")
        return False

    print(f"Converting model from: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with EXACT same architecture as training
    num_classes = len(class_names)
    model = BrainTumorCNN(num_classes)
    
    # Load trained weights
    print("Loading trained model weights...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from checkpoint with model_state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded directly from state dict")
    
    model.eval()
    print("Model loaded successfully!")

    # Create dummy input matching training preprocessing
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Convert to ONNX
    onnx_path = os.path.join(output_dir, "brain_tumor_model.onnx")
    
    print("Converting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f" ONNX model saved to: {onnx_path}")
    
    # Save preprocessing parameters that match training
    try:
        from config import Config
        preprocessing_params = {
            "image_size": list(Config.IMAGE_SIZE),
            "mean": Config.IMAGENET_MEAN,
            "std": Config.IMAGENET_STD,
            "classes": class_names
        }
    except (ImportError, AttributeError):
        # Fallback preprocessing (standard ImageNet values)
        preprocessing_params = {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "classes": class_names
        }
    
    # Save preprocessing config
    preprocessing_path = os.path.join(output_dir, "preprocessing.json")
    with open(preprocessing_path, 'w') as f:
        json.dump(preprocessing_params, f, indent=2)
    
    print(f" Preprocessing config saved to: {preprocessing_path}")
    
    # Test the ONNX model
    print("Testing ONNX model...")
    import onnxruntime as ort
    
    try:
        onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Test inference
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        test_input = dummy_input.numpy()
        onnx_result = onnx_session.run([output_name], {input_name: test_input})
        
        print(" ONNX model test successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {onnx_result[0].shape}")
        print(f"   Output classes: {len(class_names)}")
        
        # Test with PyTorch for comparison
        with torch.no_grad():
            torch_result = model(dummy_input)
            
        # Compare outputs (should be very close)
        torch_output = torch_result.numpy()
        onnx_output = onnx_result[0]
        max_diff = abs(torch_output - onnx_output).max()
        
        print(f"   Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print(" ONNX conversion verified - outputs match!")
        else:
            print("  Warning: ONNX and PyTorch outputs differ slightly")
            
    except Exception as e:
        print(f" ONNX test failed: {e}")
        print("Model converted but verification failed. Check ONNX runtime installation.")
    
    print("\n" + "="*60)
    print("MODEL CONVERSION COMPLETE!")
    print("="*60)
    print(f"Files created in {output_dir}:")
    for file in os.listdir(output_dir):
        print(f"   {file}")
    
    print(f"\nYour real brain tumor model is now ready for web deployment!")
    print(f"Classes: {class_names}")
    print(f"Architecture: ResNet50 + Enhanced Classifier")
    
    return True

def verify_model_compatibility():
    """
    Verify that the model conversion will work before running it
    """
    print("Checking model conversion requirements...")
    
    # Check PyTorch
    try:
        import torch
        print(f" PyTorch {torch.__version__} found")
    except ImportError:
        print(" PyTorch not found. Install with: pip install torch torchvision")
        return False
    
    # Check ONNX Runtime  
    try:
        import onnxruntime
        print(f" ONNX Runtime {onnxruntime.__version__} found")
    except ImportError:
        print(" ONNX Runtime not found. Install with: pip install onnxruntime")
        return False
        
    # Check torchvision
    try:
        import torchvision
        print(f" torchvision {torchvision.__version__} found")
    except ImportError:
        print(" torchvision not found. Install with: pip install torchvision")
        return False
    
    return True

if __name__ == "__main__":
    print(" Brain Tumor Model Conversion Tool")
    print("="*50)
    
    if not verify_model_compatibility():
        print("\n Missing required packages. Install them first:")
        print("pip install torch torchvision onnxruntime")
        exit(1)
    
    print("\nStarting model conversion...")
    success = convert_your_model_to_onnx()
    
    if success:
        print("\n SUCCESS! model is ready for web deployment!")
        print("\nNext steps:")
        print("1. Update your API route (see instructions)")
        print("2. Install Node.js packages: npm install onnxruntime-node sharp")
        print("3. Deploy to Vercel")
    else:
        print("\n Conversion failed. Check the error messages above.")