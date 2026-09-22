# Handwritten Character Recognition MVP

A Convolutional Neural Network (CNN) built to recognize handwritten digits (0-9) from the MNIST dataset.
This project was developed as **Task 3** for the **CodeAlpha Machine Learning Internship**.

## 🚀 Project Overview

The objective of this project is to build a Minimum Viable Product (MVP) capable of classifying handwritten characters with high accuracy.

The model was trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and achieves a **98.75% Test Accuracy** in just 5 training epochs.

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.x |
| Deep Learning Framework | TensorFlow / Keras |
| Data Manipulation | NumPy |
| Data Visualization | Matplotlib |

## 🧠 Model Architecture

The core of this project is a Sequential Convolutional Neural Network designed to process spatial hierarchies:

```
Input (28x28x1)
   │
   ▼
Conv2D (32 filters, 3x3) + ReLU     → extracts edges & curves
   │
   ▼
MaxPooling2D (2x2)                  → downsamples
   │
   ▼
Conv2D (64 filters, 3x3) + ReLU     → captures complex shapes
   │
   ▼
MaxPooling2D (2x2)                  → downsamples further
   │
   ▼
Flatten                             → 2D → 1D vector
   │
   ▼
Dense (64 units, ReLU)              → feature interpretation
   │
   ▼
Dense (10 units, Softmax)           → class probabilities
```

| Setting | Value |
|---|---|
| Loss Function | `sparse_categorical_crossentropy` |
| Optimizer | `adam` |

## ⚙️ Setup & Installation

**1. Clone the repository:**

```bash
git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName
```

**2. Install the required dependencies:**

```bash
pip install tensorflow matplotlib numpy
```

## 📂 Usage & Execution Phases

The project was built and documented in a phase-by-phase approach for educational purposes:

| File | Description |
|---|---|
| `phase1.py` | Downloads and loads the MNIST dataset. Prints the data shapes. |
| `phase2.py` | Normalizes pixel values (0.0–1.0), reshapes data for CNN input, and visualizes sample images. |
| `phase3.py` | Constructs the CNN architecture and outputs `model.summary()`. |
| `phase4.py` | Compiles and trains the model for 5 epochs. Evaluates on unseen test data. |
| `phase5.py` | Runs live inference on a test image, visualizes the prediction vs. true label, and saves the model as `handwritten_digit_model.keras`. |

**To run the final inference and see the model in action:**

```bash
python phase5.py
```

## 📈 Results

| Metric | Value |
|---|---|
| Final Training Accuracy | ~99.3% |
| Final Test Accuracy | **98.75%** |

## 🤝 Acknowledgments

Special thanks to **CodeAlpha** for the internship opportunity and the hands-on experience in applied Computer Vision and Deep Learning.
