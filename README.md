![Header](https://capsule-render.vercel.app/api?type=waving&color=0:6366F1,100:22D3EE&height=180&section=header&text=Handwritten%20Digit%20Recognition&fontSize=35&fontColor=ffffff&animation=fadeIn)

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=22D3EE&center=true&vCenter=true&width=500&lines=CNN+trained+on+MNIST;98.75%25+Test+Accuracy;CodeAlpha+ML+Internship+Task+3" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white" />
</p>

# Handwritten Character Recognition MVP

A Convolutional Neural Network (CNN) built to recognize handwritten digits (0-9) from the MNIST dataset.
This project was developed as **Task 3** for the **CodeAlpha Machine Learning Internship**.

## 🚀 Project Overview

The objective of this project is to build a Minimum Viable Product (MVP) capable of classifying handwritten characters with high accuracy.

The model was trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and achieves a **98.75% Test Accuracy** in just 5 training epochs.

![divider](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)

## 🔄 Project Workflow

```mermaid
flowchart LR
    A["📥 phase1.py<br/>Load MNIST dataset"] --> B["🧹 phase2.py<br/>Normalize & reshape<br/>Visualize samples"]
    B --> C["🏗️ phase3.py<br/>Build CNN architecture<br/>model.summary()"]
    C --> D["🎯 phase4.py<br/>Train (5 epochs)<br/>Evaluate on test set"]
    D --> E["🔮 phase5.py<br/>Run inference<br/>Save .keras model"]

    style A fill:#6366F1,color:#fff
    style B fill:#6366F1,color:#fff
    style C fill:#22D3EE,color:#000
    style D fill:#22D3EE,color:#000
    style E fill:#10B981,color:#fff
```

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.x |
| Deep Learning Framework | TensorFlow / Keras |
| Data Manipulation | NumPy |
| Data Visualization | Matplotlib |

![divider](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)

## 🧠 Model Architecture

The core of this project is a Sequential Convolutional Neural Network designed to process spatial hierarchies:

```mermaid
flowchart TD
    Input(["Input Image<br/>28x28x1"]) --> Conv1["Conv2D<br/>32 filters, 3x3<br/>+ ReLU"]
    Conv1 --> Pool1["MaxPooling2D<br/>2x2"]
    Pool1 --> Conv2["Conv2D<br/>64 filters, 3x3<br/>+ ReLU"]
    Conv2 --> Pool2["MaxPooling2D<br/>2x2"]
    Pool2 --> Flat["Flatten<br/>2D → 1D vector"]
    Flat --> Dense1["Dense<br/>64 units, ReLU"]
    Dense1 --> Dense2["Dense (Output)<br/>10 units, Softmax"]
    Dense2 --> Output(["Predicted Digit<br/>0-9"])

    style Input fill:#374151,color:#fff
    style Conv1 fill:#6366F1,color:#fff
    style Pool1 fill:#818CF8,color:#000
    style Conv2 fill:#6366F1,color:#fff
    style Pool2 fill:#818CF8,color:#000
    style Flat fill:#22D3EE,color:#000
    style Dense1 fill:#10B981,color:#fff
    style Dense2 fill:#10B981,color:#fff
    style Output fill:#374151,color:#fff
```

| Setting | Value |
|---|---|
| Loss Function | `sparse_categorical_crossentropy` |
| Optimizer | `adam` |

![divider](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)

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

![divider](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)

## 📈 Results

| Metric | Value |
|---|---|
| Final Training Accuracy | ~99.3% |
| Final Test Accuracy | **98.75%** |

## 🤝 Acknowledgments

Special thanks to **CodeAlpha** for the internship opportunity and the hands-on experience in applied Computer Vision and Deep Learning.

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:6366F1,100:22D3EE&height=100&section=footer)
