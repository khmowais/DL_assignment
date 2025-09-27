# Deep Learning Assignment 1: Facial Affect Recognition

## Project Overview

This project implements a deep learning pipeline for facial affect recognition, including **facial expression classification** (8 classes) and **dimensional emotion regression** (valence and arousal). The system uses multiple CNN architectures with transfer learning and provides a performance comparison.

---

## Architecture & Design

### Modular Pipeline Structure

The solution follows a modular design:

```

Modular Pipeline
├── Cell 1: Configuration & Imports
├── Cell 2: Data Loader
├── Cell 3: Data Analyzer
├── Cell 4: Data Generator
├── Cell 5: Model Factory
├── Cell 6: Model Trainer
├── Cell 7: Model Evaluator
└── Cell 8: Main Pipeline

```

### Key Features

- **Multi-task Learning**: Simultaneous classification and regression
- **Multiple Architectures**: ResNet50, EfficientNetB0, Custom CNN
- **Transfer Learning**: Pre-trained models with fine-tuning
- **Data Augmentation**: Real-time during training
- **Comprehensive Evaluation**: Multiple metrics for both tasks
- **RAM Optimization**: Streaming support for large datasets

---

## Dataset Description

### Dataset Structure
```

Dataset/
├── annotations/          # 3999 .npy files
│   ├── [prefix]_exp.npy  # Expression labels (0-7)
│   ├── [prefix]_val.npy  # Valence values (-1 to +1)
│   ├── [prefix]_aro.npy  # Arousal values (-1 to +1)
│   └── [prefix]_lnd.npy  # Facial landmarks
└── images/              # 3999 image files
└── [prefix].jpg     # Cropped face images (224×224×3)

````

### Expression Labels
| ID | Expression | Samples | Percentage |
|----|------------|---------|------------|
| 0  | Neutral | 500 | 12.5% |
| 1  | Happy | 500 | 12.5% |
| 2  | Sad | 500 | 12.5% |
| 3  | Surprise | 500 | 12.5% |
| 4  | Fear | 500 | 12.5% |
| 5  | Disgust | 500 | 12.5% |
| 6  | Anger | 500 | 12.5% |
| 7  | Contempt | 499 | 12.5% |

### Continuous Dimensions
- **Valence**: Range -0.987 to 0.982 (Mean: -0.192)
- **Arousal**: Range -0.667 to 0.984 (Mean: 0.354)

---

## Model Architectures

### 1. ResNet50-Based Model
```python
Input (224, 224, 3)
↓
ResNet50 Base (frozen weights)
↓
Global Average Pooling
↓
Dense(512) → ReLU → BatchNorm → Dropout(0.5)
↓
Dense(256) → ReLU → Dropout(0.3)
↓
Multi-Output:
- Expression: Dense(8) → Softmax
- Valence: Dense(1) → Tanh  
- Arousal: Dense(1) → Tanh
````

### 2. EfficientNetB0-Based Model

```python
Input (224, 224, 3)
↓
EfficientNetB0 Base (frozen weights)
↓
Global Average Pooling
↓
Dense(512) → ReLU → BatchNorm → Dropout(0.5)
↓
Multi-Output (same as ResNet50)
```

### 3. Custom CNN Architecture

```python
Input (224, 224, 3)
↓
Conv2D(32)→BN→ReLU→MaxPool
↓
Conv2D(64)→BN→ReLU→MaxPool
↓
Conv2D(128)→BN→ReLU→MaxPool
↓
Conv2D(256)→BN→ReLU→GlobalAvgPool
↓
Dense(512)→ReLU→Dropout(0.5)
↓
Dense(256)→ReLU→Dropout(0.3)
↓
Multi-Output (same as above)
```

---

## Training Configuration

### Hyperparameters

```python
Config(
    IMG_SIZE=(224, 224),
    BATCH_SIZE=32,
    EPOCHS=30,
    FINE_TUNE_EPOCHS=10,
    VALIDATION_SPLIT=0.15,
    TEST_SPLIT=0.15,
    LEARNING_RATE=0.001,
    FINE_TUNE_LR=0.0001
)
```

### Two-Phase Training Strategy

1. **Phase 1 - Feature Extraction**: Train custom head only with a learning rate of 0.001.
2. **Phase 2 - Fine-Tuning**: Unfreeze base model and train entire model with a learning rate of 0.0001.

### Loss Function (Multi-Task)

```python
Total Loss = 1.0 × Expression_Loss + 0.5 × Valence_Loss + 0.5 × Arousal_Loss

Where:
- Expression_Loss = SparseCategoricalCrossentropy
- Valence_Loss = MeanSquaredError
- Arousal_Loss = MeanSquaredError
```

---

## Evaluation Metrics

### Expression Classification

* **Accuracy**: Overall classification accuracy
* **Top-K Accuracy**: Accuracy in top K predictions
* **Confusion Matrix**: Per-class performance

### Valence/Arousal Regression

* **MSE (Mean Squared Error)**: Overall error magnitude
* **MAE (Mean Absolute Error)**: Average absolute error
* **RMSE (Root MSE)**: Error in original units
* **Correlation**: Linear relationship with ground truth

---

## Implementation Details

### Data Loading & Preprocessing

* Batch loading for memory efficiency
* Real-time data augmentation
* RAM optimization with streaming mode
* Stratified train/validation/test splits

### Data Augmentation

```python
AugmentationParams(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest'
)
```

### Model Training Features

```python
TrainingFeatures(
    early_stopping=patience(10),
    learning_rate_reduction=factor(0.5),
    model_checkpointing=True,
    multi-worker support=True
)
```

---

## Performance Results

### Sample Training Output (ResNet50 - Epoch 1-2)

```
Epoch 1/30:
- Expression Accuracy: 0.1463 → Val: 0.1067
- Valence MAE: 0.5617 → Val: 0.5885
- Arousal MAE: 0.5142 → Val: 0.4633

Epoch 2/30:
- Expression Accuracy: 0.1725 (improving)
- Valence MAE: 0.5107 (improving)
- Arousal MAE: 0.4201 (improving)
```

---

## Output Files Generated

### Model Files

* `best_ResNet50.h5` - Best ResNet50 model weights
* `best_EfficientNetB0.h5` - Best EfficientNet model weights
* `best_CustomCNN.h5` - Best custom CNN model weights

### Analysis Files

* `dataset_analysis.png` - Dataset distribution visualization
* `model_comparison.png` - Performance comparison chart
* `model_comparison_results.csv` - Quantitative results
* `*_training_history.png` - Training curves for each model
* `*_confusion_matrix.png` - Confusion matrices

---

## Technical Specifications

### Software Requirements

```python
Dependencies:
- TensorFlow 2.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Pillow (PIL)
```

### Hardware Requirements

* **Minimum**: 8GB RAM, GPU with 4GB VRAM
* **Recommended**: 12GB+ RAM, GPU with 8GB+ VRAM

---

## Usage Instructions

### Quick Start

1. **Run Cell 1**: Import dependencies and set configuration
2. **Run Cells 2-7**: Define pipeline components
3. **Run Cell 8**: Execute the training pipeline

### Customization Options

```python
# Modify these in Cell 1 (Config class):
config.MAX_SAMPLES = 1000
config.USE_LIGHT_MODELS = True
config.STREAMING_MODE = True
config.EPOCHS = 20
```

---

## Discussion & Analysis

### Expected Challenges

1. **Class Imbalance**: Some expressions may be underrepresented
2. **Regression Difficulty**: Valence/arousal prediction is challenging
3. **Overfitting Risk**: Limited dataset size for deep models

### Performance Expectations

* **Expression Classification**: 50-70% accuracy
* **Valence Regression**: MAE of 0.2-0.3
* **Arousal Regression**: MAE of 0.2-0.3

### Model Selection Rationale

* **ResNet50**: Strong baseline, good feature extraction
- **EfficientNet**: Parameter efficiency, good accuracy
- **Custom CNN**: Control over architecture, interpretability

---

## Future Improvements

### Immediate Enhancements
1. **Advanced Augmentation**: Implement MixUp, CutMix, and AutoAugment
2. **Class Balancing**: Use focal loss and weighted sampling
3. **Architecture Search**: Neural Architecture Search (NAS)
4. **Ensemble Methods**: Combine models using averaging or stacking

### Advanced Features
1. **Attention Mechanisms**: Explore self-attention and spatial attention
2. **Temporal Modeling**: Process video sequences for dynamic emotion changes
3. **Multi-modal Fusion**: Combine with audio and physiological signals
4. **Uncertainty Quantification**: Explore Bayesian Neural Networks for model uncertainty

### Deployment Considerations
1. **Model Compression**: Use quantization and pruning techniques
2. **Edge Deployment**: Consider TensorFlow Lite or ONNX for deployment on mobile/edge devices
3. **API Development**: Develop a RESTful API for real-time inference
4. **Monitoring**: Implement performance tracking and drift detection

---

## References & Citations

### Key Papers
1. Goodfellow et al. - "Challenges in representation learning: A report on the black box"
2. Russakovsky et al. - "ImageNet Large Scale Visual Recognition Challenge"
3. Tan & Le - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
4. He et al. - "Deep Residual Learning for Image Recognition"

### Libraries & Frameworks
- TensorFlow/Keras: Model development
- Scikit-learn: Evaluation metrics
- Matplotlib/Seaborn: Visualization
- NumPy/Pandas: Data processing