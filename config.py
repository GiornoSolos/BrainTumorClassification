#!/usr/bin/env python
"""
Configuration file for brain tumor detection project
"""

import os
from pathlib import Path

class Config:
    """Configuration settings for the brain tumor detection project"""
    
    # Base paths - users can modify these
    BASE_DATA_DIR = os.getenv('BRAIN_TUMOR_DATA_DIR', './data')
    
    # Dataset paths
    TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, 'Training')
    TEST_DATA_DIR = os.path.join(BASE_DATA_DIR, 'Testing')
    
    # Model paths
    MODEL_DIR = './models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_brain_tumor_model.pth')
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_model_final.pth')
    
    # Results paths
    RESULTS_DIR = './results'
    CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.0001
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        dirs = [Config.MODEL_DIR, Config.RESULTS_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @staticmethod
    def get_data_path():
        """Get the data path with user guidance"""
        if not os.path.exists(Config.TRAIN_DATA_DIR):
            print("Data directory not found!")
            print("Please either:")
            print("1. Create a 'data' folder in your project directory")
            print("2. Set the BRAIN_TUMOR_DATA_DIR environment variable")
            print("3. Update the BASE_DATA_DIR in config.py")
            print(f"Looking for: {Config.TRAIN_DATA_DIR}")
            return None
        return Config.TRAIN_DATA_DIR

def setup_project():
    """Initial project setup"""
    Config.create_directories()
    
    print("Brain Tumor Detection Project Setup")
    print("=" * 40)
    print(f"Training data: {Config.TRAIN_DATA_DIR}")
    print(f"Test data: {Config.TEST_DATA_DIR}")
    print(f"Models will be saved to: {Config.MODEL_DIR}")
    print(f"Results will be saved to: {Config.RESULTS_DIR}")
    
    # Check if data exists
    if not os.path.exists(Config.TRAIN_DATA_DIR):
        print("\n⚠️ Dataset not found!")
        print("Please place your brain tumor dataset in:")
        print(f"  {Config.BASE_DATA_DIR}/")
        print("Expected structure:")
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
        return False
    
    print("\n✅ Setup complete!")
    return True

if __name__ == "__main__":
    setup_project()
