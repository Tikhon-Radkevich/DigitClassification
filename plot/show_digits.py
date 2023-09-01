import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select six random indices from the test set
indices = np.random.randint(0, len(x_test), size=6)

# Extract the corresponding digits from the test set
digits = x_test[indices]

# Get the labels for the selected digits
labels = y_test[indices]

# Print the digit matrix with beautiful spacing
separator = " "
print(np.array2string(digits[0], separator=separator, max_line_width=200))
print(labels[0])

# Create a 2x3 subplot for the digits
fig = make_subplots(rows=2, cols=3)

# Add each digit to the subplot as a heatmap, along with its label
for i in range(6):
    digit = digits[i][::-1]
    heatmap = go.Heatmap(z=digit, colorscale='gray')
    fig.add_trace(heatmap, row=i // 3 + 1, col=i % 3 + 1)
    fig.add_annotation(
        dict(text=f"Digit {labels[i]}", xref="x"+str(i+1), yref="y"+str(i+1), showarrow=False, font=dict(color="white")))

# Set the axis labels and title
fig.update_layout(xaxis=dict(title='Column'), yaxis=dict(title='Row'), title='Digits Heatmap')

# Show the figure
fig.show()
