import numpy as np

from predict import get_digit


digit = input("Print matrix: ")
digit = np.array(digit)

print(f"Digit: {get_digit(digit)}")