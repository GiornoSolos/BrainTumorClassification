#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim

# UPDATE: Change this to your brain tumor dataset path
data_dir = r'C:\Users\Administrator\Downloads\brain_tumor_dataset'  

# MODIFIED: Enhanced transforms for medical imaging
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Standard for medical imaging
    transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability for medical data
    transforms.RandomRotation(10),  # Reduced rotation for brain scans
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Slight translation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle adjustments for MRI
    transforms.ToTensor(),           
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset - brain tumor classes will be automatically detected
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Print detected classes
print(f"Detected brain tumor classes: {full_dataset.classes}")
print(f"Number of classes: {len(full_dataset.classes)}")
print(f"Total images: {len(full_dataset)}")

# Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transform to validation set
val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for medical data
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ENHANCED: Advanced CNN model for medical imaging
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        # Use ResNet50 instead of ResNet18 for better performance on medical data
        self.base_model = models.resnet50(pretrained=True)
        
        # Unfreeze more layers for medical imaging fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two ResNet blocks for better medical feature learning
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
            
        # Enhanced classifier with dropout for regularization
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

# ENHANCED: Medical imaging preprocessing functions
def enhance_medical_image(image_path):
    """Enhanced preprocessing specifically for brain MRI images"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Convert back to RGB for model compatibility
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def is_medical_outlier(image, threshold_low=10, threshold_high=240):
    """Detect outliers in medical images"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    avg_pixel_value = np.mean(gray)
    return avg_pixel_value < threshold_low or avg_pixel_value > threshold_high

# ENHANCED: Medical image quality assessment
def assess_medical_image_quality(data_dir, classes):
    """Assess and enhance medical image quality"""
    outliers = {cls: [] for cls in classes}
    
    for cls in classes:
        img_dir = os.path.join(data_dir, cls)
        if not os.path.exists(img_dir):
            print(f"Warning: Directory {img_dir} not found")
            continue
            
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for idx, img_file in enumerate(img_files):
            img_path = os.path.join(img_dir, img_file)
            image = cv2.imread(img_path)
            if image is not None and is_medical_outlier(image):
                outliers[cls].append((idx, img_path))
    
    # Enhance outlier images
    for cls, img_data in outliers.items():
        if img_data:
            print(f"Enhancing {len(img_data)} outlier images in class '{cls}'")
            for idx, img_path in img_data:
                enhanced = enhance_medical_image(img_path)
                if enhanced is not None:
                    cv2.imwrite(img_path, enhanced)

# Run image quality assessment and enhancement
print("Assessing and enhancing medical image quality...")
assess_medical_image_quality(data_dir, full_dataset.classes)

# Initialize the enhanced model
num_classes = len(full_dataset.classes)
model = BrainTumorCNN(num_classes)

# ENHANCED: Medical imaging optimized training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # AdamW for medical data

# Learning rate scheduler for medical imaging
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# ENHANCED: Training loop with medical imaging best practices
num_epochs = 25  # More epochs for medical data
best_val_accuracy = 0.0

print("Starting brain tumor classification training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    epoch_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_loss_avg = val_loss / len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_accuracy)
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_brain_tumor_model.pth')
        print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2f}%")
    print(f"Best Val Acc: {best_val_accuracy:.2f}%")
    print("-" * 50)

# Save final model
torch.save(model.state_dict(), 'brain_tumor_model_final.pth')
print(f"Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
print("Models saved: 'best_brain_tumor_model.pth' and 'brain_tumor_model_final.pth'")