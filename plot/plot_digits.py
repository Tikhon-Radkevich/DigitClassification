import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_mnist_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test


def select_random_digits(data, labels, num_digits=6):
    # Select random indices from the dataset
    indices = np.random.randint(0, len(data), size=num_digits)
    # Extract the corresponding digits and labels
    selected_digits = data[indices]
    selected_labels = labels[indices]
    return selected_digits, selected_labels


def print_digit_with_spacing(digit):
    # Print the digit matrix with spacing
    separator = " "
    print(np.array2string(digit, separator=separator, max_line_width=200))


def create_heatmap(digit):
    # Create a heatmap for a digit
    digit = digit[::-1]  # Reverse the digit for proper display
    heatmap = go.Heatmap(z=digit, colorscale='gray')
    return heatmap


def create_digit_subplot(digits, labels):
    # Create a subplot for displaying digits
    fig = make_subplots(rows=2, cols=3)

    for i in range(len(digits)):
        digit = digits[i]
        heatmap = create_heatmap(digit)
        fig.add_trace(heatmap, row=i // 3 + 1, col=i % 3 + 1)
        fig.update_xaxes(title_text=f"Digit {labels[i]}", row=i // 3 + 1, col=i % 3 + 1)

    # Set axis labels and title
    fig.update_layout(xaxis=dict(title='Column'), yaxis=dict(title='Row'), title='Digits Heatmap',
                      template="plotly_dark")
    return fig


def main():
    x_test, y_test = load_mnist_data()
    selected_digits, selected_labels = select_random_digits(x_test, y_test)

    for i in range(len(selected_digits)):
        print_digit_with_spacing(selected_digits[i])
        print(selected_labels[i])

    heatmap_fig = create_digit_subplot(selected_digits, selected_labels)
    heatmap_fig.show()


if __name__ == "__main__":
    main()
