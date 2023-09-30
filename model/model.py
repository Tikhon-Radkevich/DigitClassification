from keras.src.saving import saving_api
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import plotly.graph_objs as go


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # scaling for future app
    bordering_value = 0.45
    x_train = np.where(x_train > bordering_value, 1.0, x_train)
    x_test = np.where(x_test > bordering_value, 1.0, x_test)
    x_train = np.where(x_train < bordering_value, 0.0, x_train)
    x_test = np.where(x_test < bordering_value, 0.0, x_test)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def create_mnist_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.15),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.15),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax")
    ])
    return model


def train_mnist_model(model, x_train, y_train):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=64, epochs=8, validation_split=0.2)
    return history


def evaluate_mnist_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc}")  # max test accuracy: 0.9925


def save_mnist_model(model, filename):
    saving_api.save_model(model, filename)


def plot_training_curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history["accuracy"], name="Training Accuracy"))
    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history["val_accuracy"], name="Validation Accuracy"))
    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history["loss"], name="Training Loss"))
    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history.history["val_loss"], name="Validation Loss"))
    fig.update_layout(title="Training and Validation Curves",
                      xaxis_title="Epoch",
                      yaxis_title="Value",
                      template="plotly_dark")
    fig.show()


def main():
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = create_mnist_model()
    model.summary()
    history = train_mnist_model(model, x_train, y_train)
    evaluate_mnist_model(model, x_test, y_test)
    save_mnist_model(model, "digit_classification_model.h5")
    plot_training_curves(history)


if __name__ == "__main__":
    main()
