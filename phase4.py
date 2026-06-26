import tensorflow as tf

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

# 1. Compile the model
# - Optimizer: 'adam' is a smart algorithm that automatically adjusts the learning rate to train efficiently.
# - Loss function: 'sparse_categorical_crossentropy' is the standard for multi-class classification 
#   when labels are integers (e.g., 5) instead of arrays (e.g., [0,0,0,0,0,1,0,0,0,0]).
# - Metrics: We want to track the 'accuracy' (percentage of correct guesses).
print("\nCompiling the model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 2. Train the model
# We call `model.fit()` to start the training process.
# - epochs=5: The model will look at the entire training dataset 5 separate times.
# - validation_data: After every epoch, the model is tested on unseen data to ensure it is actually learning.
print("Starting training for 5 epochs... This might take a minute.")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 3. Evaluate the model
# Finally, we do a formal evaluation on the test dataset to get our final score.
print("\nEvaluating final model on the test dataset...")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
