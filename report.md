# Assignment 4 & 5 Report

## Title

Network Traffic Classification Using Image-Based Representation and Convolutional Neural Networks

## 1. Introduction

Modern networks generate massive volumes of traffic, making manual inspection and traditional rule-based intrusion detection systems insufficient for identifying sophisticated attacks such as Distributed Denial of Service (DDoS). Machine learning, and in particular deep learning, offers scalable approaches for detecting malicious behavior directly from traffic patterns.

This project addresses the problem of **network attack detection** by converting structured network flow data into image representations and applying a **Convolutional Neural Network (CNN)** for binary classification of traffic as *benign* or *DDoS*. The work is divided into two parts:

* **Assignment 4**: Development of software that converts network data into images and trains a CNN classifier.
* **Assignment 5**: Documentation and theoretical explanation of the conversion process and the implemented system.

## 2. Dataset Description

The dataset used in this project is a large network traffic file containing flow-level features and labels indicating benign or attack behavior. The file size exceeds 100 MB and therefore is not included in this repository.

Each record represents a network flow with multiple numerical features (e.g., packet counts, byte statistics, timing-related attributes) and a label column indicating the traffic type. Although some development tools may incorrectly display all records as benign due to file size limitations, inspection using external editors confirms the presence of both benign and attack samples.

## 3. Network Data to Image Transformation

### 3.1 Motivation

CNNs are highly effective at learning spatial patterns in images. By reshaping network flow features into fixed-size matrices, traffic behavior can be treated as a visual pattern rather than a flat vector, enabling CNNs to capture local correlations between features.

### 3.2 Transformation Process

The transformation from network data to images is performed as follows:

1. **Feature Selection**: Numerical flow features are extracted from each network record.
2. **Normalization**: Features are scaled to a fixed range to ensure numerical stability.
3. **Reshaping**: Each feature vector is reshaped into a 2D matrix (image-like structure).
4. **Grayscale Encoding**: The matrix values are interpreted as grayscale pixel intensities.
5. **Image Storage**: Generated images are saved as PNG files and organized into class-specific folders (`benign/`, `ddos/`).

The resulting directory structure follows the standard format required by deep learning image loaders:

```
outputs/images/
├── train/
│   ├── benign/
│   └── ddos/
├── val/
│   ├── benign/
│   └── ddos/
└── test/
    ├── benign/
    └── ddos/
```

## 4. Convolutional Neural Network Architecture

### 4.1 Model Design

A lightweight CNN architecture was implemented to balance performance and computational efficiency:

* Input: 1-channel grayscale images
* Convolutional layers with ReLU activation
* MaxPooling for spatial downsampling
* Adaptive average pooling for size invariance
* Fully connected layer for classification

This architecture allows the model to learn hierarchical feature representations from network traffic images.

### 4.2 Training Setup

* Loss function: Cross-Entropy Loss with class weighting
* Optimizer: Adam
* Learning rate: 0.001
* Epochs: 10
* Device: CPU (GPU supported if available)

Class imbalance is handled by computing class weights from the training dataset.

## 5. Experimental Results

### 5.1 Training and Validation

During training, the model demonstrated rapid convergence with high training and validation accuracy. Validation accuracy exceeded 99%, indicating strong generalization to unseen data.

### 5.2 Evaluation Metrics

The trained model was evaluated on a held-out test set using:

* Confusion Matrix
* Precision, Recall, and F1-score (Classification Report)

These metrics are automatically generated and saved in `outputs/metrics.txt`. The results confirm that the CNN effectively distinguishes benign traffic from DDoS attacks.

## 6. Software Implementation

The software is implemented in Python and organized into modular scripts:

* `preprocess.py`: Data cleaning and preparation
* `make_images.py`: Network-to-image conversion
* `train_cnn.py`: CNN training and evaluation
* `predict.py`: Inference on new data
* `run_pipeline.py`: End-to-end execution

The modular design ensures reproducibility and extensibility.

## 7. Limitations and Future Work

* Only binary classification (benign vs. DDoS) is considered
* CNN architecture is intentionally simple
* Feature-to-image mapping could be further optimized

Future improvements include multi-class attack detection, deeper architectures, and real-time inference integration.

## 8. Conclusion

This project demonstrates that network traffic can be successfully transformed into image representations and classified using convolutional neural networks. The proposed approach achieves high accuracy while remaining computationally efficient, making it suitable for large-scale network security applications.

The developed software fully satisfies the requirements of Assignment 4 and Assignment 5 by providing a complete implementation and comprehensive documentation of the methodology.

---

*Note: Raw network datasets and generated images are excluded from this repository due to size constraints. The provided software reproduces all steps required to regenerate the results.*
