import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- Phase 1: Data Loading ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# --- Phase 2: Data Preprocessing & Exploration ---

# 1. Normalize the pixel values
# Image pixels range from 0 to 255. We divide by 255 to scale them to a 0.0 - 1.0 range.
# Neural networks learn much better and faster when inputs are small, scaled numbers.
x_train = x_train / 255.0
x_test = x_test / 255.0
print("Pixel values normalized to range 0.0 - 1.0.")

# 2. Reshape the image data
# A CNN in Keras expects the input to have 4 dimensions: (number_of_images, height, width, channels).
# Grayscale images have 1 color channel (RGB would have 3). 
# Our current shape is (60000, 28, 28). We need to reshape it to (60000, 28, 28, 1).
# We use -1 for the first dimension to tell numpy to automatically figure out the number of images.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"Reshaped x_train for CNN: {x_train.shape}")
print(f"Reshaped x_test for CNN: {x_test.shape}")

# 3. Plot sample images
# Let's visualize the first 5 images from our dataset along with their labels
plt.figure(figsize=(12, 3)) # Create a wide figure

for i in range(5):
    # Create subplots (1 row, 5 columns, index i+1)
    plt.subplot(1, 5, i + 1) 
    
    # Matplotlib expects a 2D array (28, 28) for grayscale plotting, 
    # so we briefly reshape it back from (28, 28, 1) just for visualization.
    img_for_display = x_train[i].reshape(28, 28)
    
    # Plot the image using a grayscale color map
    plt.imshow(img_for_display, cmap='gray') 
    
    # Set the title of each subplot to the correct label (digit)
    plt.title(f"Label: {y_train[i]}") 
    
    # Turn off the axis ticks for a cleaner look
    plt.axis('off') 

plt.tight_layout() # Automatically adjust spacing so titles don't overlap
print("Opening a window to display sample images... Please close the image window when done.")
plt.show() # Display the actual plot UI
