# Handwritten Character Recognition MVP

A Convolutional Neural Network (CNN) built to recognize handwritten digits (0-9) from the MNIST dataset. 
This project was developed as **Task 3** for the **CodeAlpha Machine Learning Internship**.

## 🚀 Project Overview

The objective of this project is to build a Minimum Viable Product (MVP) capable of classifying handwritten characters with high accuracy. 

The model was trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and achieves a **98.75% Test Accuracy** in just 5 training epochs.

## 🛠️ Tech Stack

*   **Language:** Python 3.x
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Data Manipulation:** NumPy
*   **Data Visualization:** Matplotlib

## 🧠 Model Architecture

The core of this project is a Sequential Convolutional Neural Network designed to process spatial hierarchies:

1.  **Conv2D Layer (32 filters, 3x3):** Extracts fundamental features (edges, curves).
2.  **MaxPooling2D Layer (2x2):** Downsamples the spatial dimensions.
3.  **Conv2D Layer (64 filters, 3x3):** Captures more complex, combined shapes.
4.  **MaxPooling2D Layer (2x2):** Further downsampling.
5.  **Flatten Layer:** Converts the 2D feature map into a 1D vector.
6.  **Dense Layer (64 neurons, ReLU):** Fully connected layer for feature interpretation.
7.  **Output Dense Layer (10 neurons, Softmax):** Outputs a probability distribution across the 10 digit classes.

*Loss Function:* `sparse_categorical_crossentropy`
*Optimizer:* `adam`

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install tensorflow matplotlib numpy
    ```

## 📂 Usage & Execution Phases

The project was built and documented in a phase-by-phase approach for educational purposes:

*   **`phase1.py`**: Downloads and loads the MNIST dataset. Prints the data shapes.
*   **`phase2.py`**: Normalizes the pixel values (0.0 to 1.0), reshapes the data for CNN input, and visualizes sample images using Matplotlib.
*   **`phase3.py`**: Constructs the CNN architecture and outputs the `model.summary()`.
*   **`phase4.py`**: Compiles and trains the model for 5 epochs. Evaluates the model on unseen test data.
*   **`phase5.py`**: Runs a live inference on a single test image, visualizes the prediction alongside the true label, and saves the trained model as `handwritten_digit_model.keras`.

**To run the final inference and see the model in action:**
```bash
python phase5.py
```

## 📈 Results

*   **Final Training Accuracy:** ~99.3%
*   **Final Test Accuracy:** 98.75%

## 🤝 Acknowledgments

Special thanks to **CodeAlpha** for the internship opportunity and the hands-on experience in applied Computer Vision and Deep Learning.
