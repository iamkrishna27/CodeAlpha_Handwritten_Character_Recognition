import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- Phase 1 & 2: Data Loading and Preprocessing ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# --- Phase 3: Building the CNN Architecture ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# --- Phase 4: Compiling & Training the Model ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# We will train for just 1 epoch here so the script runs quickly,
# since we've already proven the model can achieve high accuracy.
print("Quickly training the model (1 epoch) so we can make a prediction...")
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

# --- Phase 5: Inference, Visualization, and Saving ---

print("\n--- Running Phase 5: Inference and Saving ---")

# 1. Select a single image from the test dataset
# We'll pick the image at index 15. Feel free to change this number (0 to 9999).
image_index = 15
test_image = x_test[image_index]
true_label = y_test[image_index]

# 2. Make a prediction (Inference)
# The model expects a "batch" of images, even if it's just one. 
# We use np.expand_dims to change the shape from (28, 28, 1) to (1, 28, 28, 1).
input_image = np.expand_dims(test_image, axis=0) 

# Get the predictions. This returns an array of 10 probabilities.
predictions_array = model.predict(input_image)

# np.argmax finds the index of the highest probability. This index IS our predicted digit!
predicted_label = np.argmax(predictions_array[0])
print(f"The model predicted the number: {predicted_label}")
print(f"The actual number is: {true_label}")

# 3. Visualize the image and the prediction
plt.figure(figsize=(4, 4))
plt.imshow(test_image.reshape(28, 28), cmap='gray') # Reshape back to 28x28 for matplotlib
plt.title(f"AI Prediction: {predicted_label} | True Label: {true_label}")
plt.axis('off')

print("Opening image visualization window... Close it to proceed to saving.")
plt.show()

# 4. Save the trained model
# We save the entire model (architecture, weights, and compilation info) into a file.
# '.keras' is the modern standard format for Keras models.
model_filename = "handwritten_digit_model.keras"
model.save(model_filename)
print(f"\nModel successfully saved to disk as '{model_filename}'")
print(f"To use it later without retraining, you can run: model = tf.keras.models.load_model('{model_filename}')")
