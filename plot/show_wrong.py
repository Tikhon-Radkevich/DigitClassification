from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

from show_tools import show_digits


def load_mnist_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_test, y_test


def preprocess_data(data):
    # Preprocess the data by scaling it to the range [0, 1]
    return data.astype('float32') / 255.


def find_wrong_predictions(model, x_test, y_test):
    # Find data where the model makes wrong predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    wrong_indices = np.where(y_pred_classes != y_test)[0]
    wrong_indices = np.random.choice(wrong_indices, size=6)
    wrong_digits = x_test[wrong_indices]
    labels = y_test[wrong_indices]
    wrong = y_pred_classes[wrong_indices]
    return wrong_digits, labels, wrong


def main():
    x_test, y_test = load_mnist_data()
    x_test = preprocess_data(x_test)

    model = load_model("../data/digit_classification_model.h5")

    wrong_digits, labels, wrong = find_wrong_predictions(model, x_test, y_test)

    show_digits(wrong_digits, labels, wrong)


if __name__ == "__main__":
    main()
