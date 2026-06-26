import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the MNIST dataset from Keras
# The MNIST dataset contains 70,000 28x28 grayscale images of handwritten digits (0-9).
# It's split into 60,000 images for training and 10,000 for testing.
print("Downloading and loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Print the shapes of the data to confirm successful loading
# The shape will tell us (number_of_images, width, height)
# x_train contains the images, y_train contains the true labels (the correct digit)
print(f"Training images shape (x_train): {x_train.shape}")
print(f"Training labels shape (y_train): {y_train.shape}")
print(f"Testing images shape (x_test): {x_test.shape}")
print(f"Testing labels shape (y_test): {y_test.shape}")
