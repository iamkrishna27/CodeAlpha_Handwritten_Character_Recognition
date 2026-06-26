import tensorflow as tf
import numpy as np

# --- Phase 1 & 2: Data Loading and Preprocessing ---
# (Shortened for brevity, but still doing the necessary work)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# --- Phase 3: Building the CNN Architecture ---

# Initialize a Sequential model (a linear stack of layers)
model = tf.keras.models.Sequential([
    
    # 1. First Convolutional Layer (Conv2D)
    # This layer slides 32 small 3x3 filters over the image. 
    # It acts like a magnifying glass looking for simple features like straight lines or edges.
    # We must specify the `input_shape` only for the very first layer.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # 2. First Max Pooling Layer (MaxPooling2D)
    # This shrinks the image by taking the maximum pixel value in every 2x2 grid.
    # It reduces the file size and helps the model focus purely on the most prominent features.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 3. Second Convolutional Layer (Conv2D)
    # Using 64 filters, this layer looks at the output of the first layer to find 
    # more complex shapes and combinations (like curves or corners).
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 4. Second Max Pooling Layer (MaxPooling2D)
    # Shrinks the data even further.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 5. Flatten Layer
    # The data is currently in a 2D grid format. 
    # We need to flatten it into a single 1D list (a long row of numbers) 
    # so we can feed it into a standard neural network layer.
    tf.keras.layers.Flatten(),
    
    # 6. Dense Layer (Hidden Layer)
    # A standard fully-connected layer with 64 neurons. 
    # It combines all the features found by the CNN to start making decisions.
    tf.keras.layers.Dense(64, activation='relu'),
    
    # 7. Output Dense Layer
    # The final layer has exactly 10 neurons, corresponding to our 10 possible digits (0-9).
    # The 'softmax' activation turns the raw outputs into percentages/probabilities.
    # The neuron with the highest percentage is the model's final prediction.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print a summary of the model architecture to the console
print("\n--- Model Architecture Summary ---")
model.summary()
