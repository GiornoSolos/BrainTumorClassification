#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

class Config:
    """Configuration class for brain tumor classification project"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Data paths
    TRAIN_DATA_DIR = DATA_DIR / "Training"
    TEST_DATA_DIR = DATA_DIR / "Testing"
    
    # Model paths
    BEST_MODEL_PATH = MODEL_DIR / "best_brain_tumor_model.pth"
    FINAL_MODEL_PATH = MODEL_DIR / "final_brain_tumor_model.pth"
    ONNX_MODEL_PATH = MODEL_DIR / "brain_tumor_model.onnx"
    
    # Results paths
    CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
    TRAINING_PLOTS_PATH = RESULTS_DIR / "training_plots.png"
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # Image preprocessing parameters (matching preprocessing.json)
    IMAGE_SIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Class names (matching preprocessing.json)
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
    NUM_CLASSES = len(CLASS_NAMES)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR,
            cls.TRAIN_DATA_DIR,
            cls.TEST_DATA_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"Created directories at: {cls.BASE_DIR}")
    
    @classmethod
    def get_data_path(cls):
        """Get the training data path, with user guidance if not found"""
        if cls.TRAIN_DATA_DIR.exists():
            return cls.TRAIN_DATA_DIR
        elif cls.DATA_DIR.exists():
            return cls.DATA_DIR
        else:
            print(f"Data directory not found at: {cls.DATA_DIR}")
            print("Please create the following structure:")
            print("  data/")
            print("  ├── Training/")
            print("  │   ├── glioma/")
            print("  │   ├── meningioma/")
            print("  │   ├── notumor/")
            print("  │   └── pituitary/")
            print("  └── Testing/")
            print("      ├── glioma/")
            print("      ├── meningioma/")
            print("      ├── notumor/")
            print("      └── pituitary/")
            return None
    
    @classmethod
    def get_model_info(cls):
        """Get model information"""
        return {
            "architecture": "ResNet50 + Enhanced Classifier",
            "num_classes": cls.NUM_CLASSES,
            "class_names": cls.CLASS_NAMES,
            "image_size": cls.IMAGE_SIZE,
            "batch_size": cls.BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("BRAIN TUMOR CLASSIFICATION CONFIG")
        print("=" * 50)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Model Directory: {cls.MODEL_DIR}")
        print(f"Results Directory: {cls.RESULTS_DIR}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Classes: {cls.CLASS_NAMES}")
        print("=" * 50)

# Optional: Auto-create directories when module is imported
if __name__ == "__main__":
    Config.create_directories()
    Config.print_config()