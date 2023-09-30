from tensorflow.keras.models import load_model
import numpy as np


class DigitClassifier:
    def __init__(self, model_path):
        self.model = model = load_model(model_path)

    def get_digit(self, digit):
        image = np.array(digit)
        image = image.reshape(1, 28, 28, 1)
        prediction = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(prediction)
        return predicted_digit
