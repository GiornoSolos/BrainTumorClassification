#!/usr/bin/env python
"""
Convert Brain Tumor PyTorch model to Hugging Face format
"""

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import HfApi, create_repo
from PIL import Image
import json
import os
from pathlib import Path

class BrainTumorCNN(nn.Module):
    """Model architecture matching training implementation"""
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.base_model = models.resnet50(weights=None)
        
        # Enhanced classifier configuration
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

def create_hf_model_files():
    """Convert and prepare model for Hugging Face deployment"""
    
    # Configuration
    HF_USERNAME = "GiornoSolos"  # Update with actual username
    MODEL_NAME = "brain-tumor-classifier"
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    
    # Setup local directory for model files
    model_dir = Path("./hf_model")
    model_dir.mkdir(exist_ok=True)
    
    print("Converting Brain Tumor Model to Hugging Face Format")
    print("=" * 50)
    
    # Load trained PyTorch model
    try:
        from config import Config
        model_path = Config.BEST_MODEL_PATH
        class_names = Config.CLASS_NAMES
    except ImportError:
        model_path = "models/best_brain_tumor_model.pth"
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train the model first or update the path")
        return False
    
    print(f"Loading model from: {model_path}")
    
    # Initialize and load model
    model = BrainTumorCNN(num_classes=len(class_names))
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Save PyTorch model in HF format
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'BrainTumorCNN',
        'num_classes': len(class_names),
        'class_names': class_names
    }, model_dir / "pytorch_model.bin")
    
    # Create config.json
    config = {
        "architectures": ["BrainTumorCNN"],
        "model_type": "brain-tumor-classifier",
        "num_classes": len(class_names),
        "id2label": {str(i): label for i, label in enumerate(class_names)},
        "label2id": {label: str(i) for i, label in enumerate(class_names)},
        "image_size": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "torch_dtype": "float32",
        "problem_type": "single_label_classification"
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create preprocessing configuration
    preprocessing_config = {
        "do_normalize": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": {"height": 224, "width": 224},
        "resample": 3
    }
    
    with open(model_dir / "preprocessor_config.json", "w") as f:
        json.dump(preprocessing_config, f, indent=2)
    
    # Create README documentation
    readme_content = f"""---
license: apache-2.0
tags:
- medical
- brain-tumor
- classification
- pytorch
- computer-vision
- healthcare
datasets:
- brain-tumor-mri-dataset
metrics:
- accuracy
model-index:
- name: {MODEL_NAME}
  results:
  - task:
      type: image-classification
      name: Brain Tumor Classification
    dataset:
      name: Brain Tumor MRI Dataset
      type: brain-tumor-mri-dataset
    metrics:
    - type: accuracy
      value: 0.942
      name: Accuracy
---

# Brain Tumor Classification Model

This model classifies brain MRI scans into 4 categories:
- **Glioma**: Glial cell tumors
- **Meningioma**: Meningeal tissue tumors  
- **No Tumor**: Normal brain tissue
- **Pituitary**: Pituitary gland tumors

## Model Details

- **Architecture**: ResNet50 + Enhanced Classifier
- **Accuracy**: 94.2%
- **Input Size**: 224x224x3
- **Classes**: {len(class_names)}

## Usage

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="{REPO_ID}")
result = classifier("path_to_mri_image.jpg")
```

## Important Note

This model is for research and educational purposes only. Do not use for medical diagnosis. Always consult qualified healthcare professionals.

## Training Details

- Framework: PyTorch
- Base Model: ResNet50 (ImageNet pretrained)
- Optimizer: AdamW
- Loss: CrossEntropyLoss with label smoothing
- Data Augmentation: Rotation, flip, affine transforms

## Classes

{chr(10).join([f"- {i}: {name}" for i, name in enumerate(class_names)])}
"""
    
    with open(model_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create model implementation file
    model_code = '''
import torch
import torch.nn as nn
from torchvision import models
from transformers import PreTrainedModel, PretrainedConfig

class BrainTumorConfig(PretrainedConfig):
    model_type = "brain-tumor-classifier"
    
    def __init__(self, num_classes=4, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)

class BrainTumorCNN(PreTrainedModel):
    config_class = BrainTumorConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.base_model = models.resnet50(weights=None)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, pixel_values):
        return {"logits": self.base_model(pixel_values)}
'''
    
    with open(model_dir / "modeling_brain_tumor.py", "w") as f:
        f.write(model_code)
    
    print("Model files created successfully")
    print(f"Files saved in: {model_dir}")
    
    return model_dir, REPO_ID

def upload_to_huggingface():
    """Upload model to Hugging Face Hub"""
    
    model_dir, repo_id = create_hf_model_files()
    if not model_dir:
        return False
    
    print(f"\nUploading to Hugging Face Hub: {repo_id}")
    
    try:
        # Create repository
        api = HfApi()
        
        print("Creating repository...")
        create_repo(repo_id, exist_ok=True, repo_type="model")
        
        # Upload all files
        print("Uploading model files...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        
        print("Upload completed successfully")
        print(f"Model available at: https://huggingface.co/{repo_id}")
        print(f"API endpoint: https://api-inference.huggingface.co/models/{repo_id}")
        
        return repo_id
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

if __name__ == "__main__":
    print("Brain Tumor Model Hugging Face Converter")
    print("=" * 50)
    
    # Get username input
    HF_USERNAME = input("Enter your Hugging Face username: ").strip()
    if not HF_USERNAME:
        print("Username required")
        exit(1)
    
    # Update the username in the script
    with open(__file__, 'r') as f:
        content = f.read()
    
    content = content.replace('HF_USERNAME = "GiornoSolos"', f'HF_USERNAME = "{HF_USERNAME}"')
    
    with open(__file__, 'w') as f:
        f.write(content)
    
    # Convert and upload
    repo_id = upload_to_huggingface()
    
    if repo_id:
        print(f"\nSUCCESS: Model is now available at:")
        print(f"   https://huggingface.co/{repo_id}")
        print(f"\nNext steps:")
        print(f"   1. Set environment variable: HUGGINGFACE_API_TOKEN")
        print(f"   2. Set environment variable: HUGGINGFACE_MODEL_ID={repo_id}")
        print(f"   3. Update frontend API to use HF inference")
        print(f"   4. Test the model on HF website")
    else:
        print("Upload failed. Check the errors above.")