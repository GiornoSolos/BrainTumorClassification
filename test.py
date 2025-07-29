#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        # Match the enhanced model architecture from training
        self.base_model = models.resnet50(weights=None)  # No pretrained weights needed for inference
        
        # Enhanced classifier matching the training model
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

def load_brain_tumor_model(model_path, num_classes):
    """Load the trained brain tumor detection model"""
    model = BrainTumorCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_brain_tumor_inference(model, test_loader, class_names, device):
    """Run inference with detailed medical metrics"""
    model.eval()
    results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, image_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results for each image
            for i in range(len(predicted)):
                pred_class = class_names[predicted[i].item()]
                confidence = probabilities[i][predicted[i].item()].item()
                results.append((image_paths[i], pred_class, confidence))
                all_predictions.append(predicted[i].item())
                all_probabilities.append(probabilities[i].cpu().numpy())
            
            # Calculate accuracy using folder names as ground truth
            labels = []
            for path in image_paths:
                folder_name = os.path.basename(os.path.dirname(path))
                if folder_name in class_names:
                    labels.append(class_names.index(folder_name))
                else:
                    # Handle unknown folder names
                    labels.append(0)  # Default to first class
                    
            labels = torch.tensor(labels).to(device)
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    
    # Generate detailed classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return results, accuracy, report, cm, all_labels, all_predictions

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Brain Tumor Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_medical_metrics(report, class_names):
    """Print detailed medical classification metrics"""
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*60)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1-score']:.3f}")
            print(f"  Support:   {metrics['support']}")
    
    print(f"\nOVERALL METRICS:")
    print(f"  Macro Avg F1:    {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
    print(f"  Overall Accuracy: {report['accuracy']:.3f}")

# Custom Dataset class for brain tumor images
class BrainTumorImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(BrainTumorImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], path

def main():
    # Configuration - UPDATE THESE PATHS
    model_path = "best_brain_tumor_model.pth"  # Use the best model
    test_data_dir = r'C:\Users\Administrator\Downloads\brain_tumor_dataset\Testing'
    
    # Brain tumor dataset typically has these classes - will auto-detect
    # Common classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
    # or ['no_tumor', 'tumor'] for binary classification
    
    # Medical imaging optimized transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Match training size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load test dataset
    test_dataset = BrainTumorImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Detected brain tumor classes: {class_names}")
    print(f"Number of test images: {len(test_dataset)}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_brain_tumor_model(model_path, num_classes)
        model.to(device)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and matches the number of classes")
        return

    # Run inference
    print("\nRunning brain tumor classification inference...")
    results, accuracy, report, cm, true_labels, predictions = run_brain_tumor_inference(
        model, test_loader, class_names, device
    )

    # Print results
    print(f"\nCLASSIFICATION RESULTS")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Print detailed medical metrics
    print_medical_metrics(report, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Print some individual predictions with confidence
    print(f"\nSAMPLE PREDICTIONS (showing first 10):")
    print("-" * 80)
    for i, (image_path, pred_class, confidence) in enumerate(results[:10]):
        filename = os.path.basename(image_path)
        true_class = os.path.basename(os.path.dirname(image_path))
        status = "CORRECT" if pred_class == true_class else "INCORRECT"
        print(f"{filename:30} | True: {true_class:12} | Pred: {pred_class:12} | Conf: {confidence:.3f} | {status}")
    
    # Performance interpretation
    print(f"\nPERFORMANCE ASSESSMENT:")
    if accuracy >= 95:
        print("EXCELLENT: Model shows high accuracy suitable for medical assistance")
    elif accuracy >= 90:
        print("GOOD: Model shows strong performance, suitable for screening")
    elif accuracy >= 85:
        print("MODERATE: Model needs improvement before clinical application")
    else:
        print("POOR: Model requires significant improvement for medical use")
    
    print(f"\nDetailed results saved and confusion matrix plotted.")
    print("This model could assist in medical screening applications.")

if __name__ == "__main__":
    main()