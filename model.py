import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import plotly.graph_objs as go

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Split the data and normalize the pixel values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

bordering_value = 0.45
x_train = np.where(x_train > bordering_value, 1.0, x_train)
x_test = np.where(x_test > bordering_value, 1.0, x_test)
x_train = np.where(x_train < bordering_value, 0.0, x_train)
x_test = np.where(x_test < bordering_value, 0.0, x_test)

# Reshape the data to add a channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Create the model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.15),
    Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.15),
    Flatten(input_shape=(28, 28, 1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)  # Test accuracy: 0.9925000071525574

# Save the model
model.save('digit_classification_model.h5')

# Plot the accuracy and loss curves for training and validation
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history['accuracy'], name='Training Accuracy'))
fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history['val_accuracy'], name='Validation Accuracy'))
fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history['loss'], name='Training Loss'))
fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history['val_loss'], name='Validation Loss'))
fig.update_layout(title='Training and Validation Curves',
                  xaxis_title='Epoch',
                  yaxis_title='Value',
                  template='plotly_dark')
fig.show()
