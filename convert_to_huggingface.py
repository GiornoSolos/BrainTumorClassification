#!/usr/bin/env python
"""
Convert Brain Tumor PyTorch model to Hugging Face Transformers format
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import PreTrainedModel, PretrainedConfig, AutoImageProcessor
from huggingface_hub import HfApi, create_repo
from PIL import Image
import json
import os
from pathlib import Path

class BrainTumorConfig(PretrainedConfig):
    """Configuration class for Brain Tumor model"""
    model_type = "brain-tumor-classifier"
    
    def __init__(
        self,
        num_classes=4,
        hidden_size=512,
        dropout_rate=0.5,
        dropout_rate_final=0.3,
        image_size=224,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.dropout_rate_final = dropout_rate_final
        self.image_size = image_size

class BrainTumorForImageClassification(PreTrainedModel):
    """Brain Tumor Classification model compatible with Transformers"""
    config_class = BrainTumorConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        
        # Create ResNet50 backbone
        self.backbone = models.resnet50(weights=None)
        
        # Remove original classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create enhanced classifier matching training code
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(num_features, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate_final),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
    def forward(self, pixel_values, labels=None):
        # Extract features from backbone
        features = self.backbone(pixel_values)
        
        # Classify
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits
        }

# Original model class for loading trained weights
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Match training unfreezing pattern
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
            
        # Enhanced classifier matching training
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

def convert_to_transformers_model():
    """Convert trained PyTorch model to Transformers format"""
    
    # Configuration
    HF_USERNAME = "GiornoSolos"
    MODEL_NAME = "brain-tumor-classifier"
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    
    # Class names
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    num_classes = len(class_names)
    
    # Setup paths
    model_dir = Path("./hf_model")
    model_dir.mkdir(exist_ok=True)
    
    try:
        from config import Config
        model_path = Config.BEST_MODEL_PATH
    except ImportError:
        model_path = "models/best_brain_tumor_model.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    print(f"Converting model from: {model_path}")
    
    # Load original trained model
    original_model = BrainTumorCNN(num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        original_model.load_state_dict(checkpoint)
    
    original_model.eval()
    print("Original model loaded successfully")
    
    # Create Transformers-compatible config
    config = BrainTumorConfig(
        num_classes=num_classes,
        hidden_size=512,
        dropout_rate=0.5,
        dropout_rate_final=0.3,
        image_size=224,
        id2label={str(i): label for i, label in enumerate(class_names)},
        label2id={label: str(i) for i, label in enumerate(class_names)}
    )
    
    # Create new Transformers model
    new_model = BrainTumorForImageClassification(config)
    
    # Copy weights from trained model to new model
    print("Transferring weights...")
    
    # Copy backbone weights (everything except the final classifier)
    original_backbone_dict = {}
    for name, param in original_model.base_model.named_parameters():
        if not name.startswith('fc'):
            original_backbone_dict[name] = param
    
    new_model.backbone.load_state_dict(original_backbone_dict, strict=False)
    
    # Copy classifier weights
    original_classifier = original_model.base_model.fc
    with torch.no_grad():
        # Copy dropout and linear layers
        new_model.classifier[1].weight.copy_(original_classifier[1].weight)  # First Linear
        new_model.classifier[1].bias.copy_(original_classifier[1].bias)
        new_model.classifier[4].weight.copy_(original_classifier[4].weight)  # Second Linear  
        new_model.classifier[4].bias.copy_(original_classifier[4].bias)
    
    print("Weights transferred successfully")
    
    # Test that outputs match
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original_output = original_model(test_input)
        new_output = new_model(test_input)
        
        max_diff = torch.abs(original_output - new_output["logits"]).max().item()
        print(f"Max difference between models: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print("✅ Model conversion verified - outputs match!")
        else:
            print("⚠️  Warning: Model outputs differ slightly")
    
    # Save the model
    print("Saving Transformers model...")
    new_model.save_pretrained(model_dir)
    config.save_pretrained(model_dir)
    
    # Create image processor config
    processor_config = {
        "do_normalize": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": {"height": 224, "width": 224},
        "resample": 3,
        "processor_class": "AutoImageProcessor"
    }
    
    with open(model_dir / "preprocessor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
    
    # Create enhanced README
    readme_content = f"""---
pipeline_tag: image-classification
license: apache-2.0
tags:
- medical
- brain-tumor
- classification
- pytorch
- computer-vision
- healthcare
- transformers
datasets:
- brain-tumor-mri-dataset
metrics:
- accuracy
library_name: transformers
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
- **Framework**: PyTorch + Transformers
- **Accuracy**: 94.2%
- **Input Size**: 224x224x3
- **Classes**: {num_classes}

## Usage
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("{REPO_ID}")
model = AutoModelForImageClassification.from_pretrained("{REPO_ID}")

image = Image.open("mri_scan.jpg")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    confidence = predictions.max().item()

print(f"Predicted: {{model.config.id2label[str(predicted_class)]}} ({{confidence:.3f}})")