from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

from show_tools import show_digits


# Find some data where the model makes wrong predictions
model = load_model("digit_classification_model.h5")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
wrong_indices = np.where(y_pred_classes != y_test)[0]
wrong_indices = np.random.choice(wrong_indices, size=6)
wrong_digits = x_test[wrong_indices]
labels = y_test[wrong_indices]
wrong = y_pred_classes[wrong_indices]

show_digits(wrong_digits, labels, wrong)