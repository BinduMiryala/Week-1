# 🌿 Plant Disease Detection System

A *Deep Learning-based* Plant Disease Detection System using *TensorFlow* and *Convolutional Neural Networks (CNNs). The model is trained on a large image dataset to classify **38 different plant diseases* with high accuracy.

![TensorFlow](https://img.shields.io/badge/Built%20With-TensorFlow-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/github/license/yourusername/plant-disease-detection?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)

---

## 🔍 Features

- *38 Plant Disease Classes*
- Over *70,000 training images*
- *Custom CNN model* with 7.8M+ parameters
- *GPU-efficient training*
- Uses image_dataset_from_directory() for flexible data loading

---

## 📁 Dataset

- *Training Samples*: 70,295
- *Validation Samples*: 17,572
- *Image Size*: 128x128 (RGB)
- *Total Classes*: 38

---

## 🧠 Model Architecture

plaintext
Conv2D -> Conv2D -> MaxPooling2D  
-> Conv2D -> Conv2D -> MaxPooling2D  
-> Conv2D -> Conv2D -> MaxPooling2D  
-> Conv2D -> Conv2D -> MaxPooling2D  
-> Conv2D -> Conv2D -> MaxPooling2D  
-> Dropout -> Flatten -> Dense(1500)  
-> Dropout -> Dense(38) (Softmax)  


- ✅ 9 Convolutional Layers  
- ✅ 5 MaxPooling Layers  
- ✅ Dropout for regularization  
- ✅ Final Dense layer with 38 softmax outputs

---

## ⚙ Training Configuration

- *Optimizer*: Adam (lr = 0.0001)  
- *Loss Function*: Categorical Crossentropy  
- *Batch Size*: 32  
- *Epochs*: 3 (can be increased for better performance)

---

## 🧑‍💻 Installation & Setup

### Prerequisites

Before running the system, ensure that you have the following installed:
- Python 3.x (You can download it from https://www.python.org/downloads/)
- Git (For cloning the repository)

### Steps to Install and Run:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/BinduMiryala/dengue-disease-prediction-system.git
    cd dengue-disease-prediction-system
    ```
2.## Installation

To install the required Python packages, run:

```bash
pip install -r requirements.txt


3. **Prepare dataset folders(if not already present)**:
    ```bash
    mkdir -p train valid
    ```
4. **Run the training script*:
    If you want to retrain the machine learning model with your own data:
    ```bash
    python train.py
    ```
---

## 📊 Results

- Accuracy improves steadily with more epochs
- Robust performance on unseen validation data
- Can be enhanced with data augmentation and fine-tuning

---

## 🚀 Future Scope

- Real-time disease prediction via webcam or mobile camera  
- Deploy as a web app using *Flask* or *Streamlit*  
- Evaluate using *confusion matrix* and *F1-score*  
- Apply *transfer learning* with pretrained CNNs (e.g., VGG, ResNet)

---

## 📝 License

This project is licensed under the *MIT License*.

---

Made with passion for precision agriculture.
