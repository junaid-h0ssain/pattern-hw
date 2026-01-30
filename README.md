# Plant Disease Classification Using Deep Learning

## Overview

This project performs a comprehensive comparative analysis of six state-of-the-art pre-trained convolutional neural networks (CNNs) for plant disease classification. The goal is to evaluate and compare the performance of different deep learning architectures on a multi-class plant disease dataset to identify the most effective model for agricultural disease detection.

## What This Project Does

The project implements transfer learning using six different CNN architectures to classify plant diseases from images. It follows a two-stage training approach:

1. **Stage 1 - Initial Training**: Train the custom classification layers while keeping the pre-trained base model frozen
2. **Stage 2 - Fine-Tuning**: Unfreeze the top layers of the base model and continue training with a lower learning rate

Each model is evaluated on multiple metrics including accuracy, precision, recall, and F1-score. The project aims to achieve greater than 98.5% validation accuracy across all models.

### Models Evaluated

- **VGG16** - 16-layer Visual Geometry Group network
- **VGG19** - Deeper 19-layer VGG variant
- **InceptionV3** - Google's Inception architecture with factorized convolutions
- **Xception** - Extreme Inception using depthwise separable convolutions
- **ResNet50** - 50-layer Residual Network with skip connections
- **DenseNet121** - Densely Connected Network with 121 layers

### Key Features

- Transfer learning from ImageNet pre-trained weights
- Two-stage training strategy (frozen base + fine-tuning)
- Advanced data augmentation (flips, rotation, zoom, contrast)
- Mixed precision training for improved performance
- Comprehensive evaluation metrics and visualization
- Automated model checkpointing and early stopping
- Learning rate reduction on plateau

## Tech Stack

### Core Framework
- **Python 3.12+** - Programming language
- **TensorFlow 2.19** - Deep learning framework
  - Keras API for model building
  - Mixed precision training support
  - GPU acceleration with CUDA

### Data Processing
- **Pandas 2.3.3+** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations
- **ImageDataGenerator** - Image preprocessing and augmentation

### Deep Learning Components
- **Pre-trained Models**: VGG16, VGG19, InceptionV3, Xception, ResNet50, DenseNet121
- **Optimizers**: Adam with adaptive learning rate
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Loss Function**: Categorical cross-entropy
- **Activation Functions**: ReLU (hidden layers), Softmax (output layer)

### Model Architecture Components
- GlobalAveragePooling2D
- Dense layers with batch normalization
- Dropout for regularization
- Custom classification head (512 -> 256 -> 128 neurons)

### Development Environment
- **uv** - Fast Python package installer and dependency manager
- **Jupyter Notebook** - Interactive development and analysis
- **IPython Kernel 7.1.0+** - Notebook execution engine
- **KaggleHub** - Dataset downloading and management

### Dataset
- **New Plant Diseases Dataset (Augmented)** from Kaggle
- Organized into train/validation/test splits
- Multi-class classification (38 plant disease categories)
- Image size: 128x128 pixels (RGB)
- Batch size: 64

## Project Structure

```
pattern/
├── comparative-analysis-pattern.ipynb  # Main analysis notebook
├── main.py                             # Entry point script
├── pyproject.toml                      # Project dependencies
└── README.md                           # This file
```

## Training Strategy

### Data Augmentation
- Horizontal and vertical flips
- Random rotation (20%)
- Random zoom (20%)
- Random contrast adjustment (20%)
- Normalization (scaling to 0-1 range)

### Training Configuration
- Initial training epochs: 10
- Fine-tuning epochs: 10
- Initial learning rate: 0.001
- Fine-tuning learning rate: 0.0001
- Early stopping patience: 5 epochs
- Learning rate reduction patience: 3 epochs

### Performance Optimizations
- Mixed precision (float16) training
- JIT compilation for model execution
- Parallel data loading with prefetching
- GPU acceleration when available

## Output Artifacts

The training process generates the following files for each model:

- `{ModelName}_best.weights.h5` - Best model weights from training
- `{ModelName}_plant_disease_model_improved.h5` - Final saved model
- `{ModelName}_y_pred_improved.npy` - Model predictions
- `{ModelName}_y_true_improved.npy` - True labels
- `{ModelName}_classification_report_improved.txt` - Detailed metrics per class
- `model_comparison_improved.csv` - Comparative performance table
- `{BestModel}_training_history.png` - Training curves visualization

## Evaluation Metrics

Each model is evaluated on:
- **Accuracy** - Overall classification accuracy
- **Top-5 Accuracy** - Percentage where true class is in top 5 predictions
- **Precision** - Weighted average across all classes
- **Recall** - Weighted average across all classes
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Per-class performance analysis

## Usage

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook comparative-analysis-pattern.ipynb
   ```

4. The notebook will:
   - Download the dataset from Kaggle
   - Train all six models
   - Generate comparison reports and visualizations
   - Save trained models and predictions

## Requirements

- Python 3.12 or higher
- TensorFlow with CUDA support (optional but recommended for GPU training)
- At least 8GB RAM (16GB+ recommended)
- GPU with CUDA support (optional but significantly faster)
- Sufficient disk space for dataset and model weights

## Results

The notebook generates comprehensive comparison tables showing:
- Model-by-model accuracy comparison
- Training and validation curves
- Per-class performance metrics
- Best performing model identification
- Models meeting the 98.5% accuracy threshold

## License

This project is provided as-is for educational and research purposes.