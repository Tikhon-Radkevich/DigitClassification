from tensorflow.keras.models import load_model
import numpy as np

model = load_model("digit_classification_model.h5")


def get_digit(digit):
    image = np.array(digit)
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction)
    return predicted_digit
