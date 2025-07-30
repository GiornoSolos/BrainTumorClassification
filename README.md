# Brain Tumor Classification

A deep learning project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNN).

## Overview

This project uses a ResNet50-based CNN to classify brain MRI scans into four categories:
- **Glioma** - A type of brain tumor
- **Meningioma** - Another type of brain tumor  
- **No Tumor** - Healthy brain scans
- **Pituitary** - Pituitary gland tumors

## Dataset

This project uses the Brain Tumor MRI Dataset available on Kaggle:
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Setup

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd BrainTumourClassification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and setup dataset**
   - Download the brain tumor dataset from Kaggle
   - Extract to your desired location
   - Update the `BASE_DATA_DIR` path in `config.py`

4. **Expected folder structure:**
   ```
   your_dataset_folder/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

## Usage

### Training
```bash
python train.py
```
- Trains a ResNet50 model on the brain tumor dataset
- Saves the best model to `models/best_brain_tumor_model.pth`
- Training time: 4-8 hours on CPU, 30-90 minutes on GPU

### Testing
```bash
python test.py
```
- Evaluates the trained model on test data
- Generates confusion matrix and detailed metrics
- Shows sample predictions with confidence scores

## Configuration

Edit `config.py` to customize:
- Dataset paths
- Model parameters
- Training settings
- Output directories

## Results

The model achieves:
- **Target Accuracy**: 90-97% on brain tumor classification
- **Medical Application**: Suitable for screening assistance
- **Classes**: 4-way classification of brain tumor types

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for complete list

## Project Structure

```
BrainTumourClassification/
├── config.py          # Configuration settings
├── train.py           # Training script
├── test.py            # Testing/evaluation script
├── requirements.txt   # Python dependencies
├── models/            # Saved model files
└── results/           # Generated results and plots
```

## Notes

- This model is for educational/research purposes
- Not intended for actual medical diagnosis
- Always consult medical professionals for health decisions
