import plotly.graph_objs as go
from plotly.subplots import make_subplots


def show_digits(digits, labels, wrong):
    # Create a 2x3 subplot for the digits
    fig = make_subplots(rows=2, cols=3)

    # Define titles for each subplot
    subplot_titles = [f"True: {labels[i]} - Predict: {wrong[i]}" for i in range(6)]

    # Add each digit to the subplot as a heatmap, along with its label and title
    for i in range(6):
        digit = digits[i][::-1]
        heatmap = go.Heatmap(z=digit, colorscale='gray')
        fig.add_trace(heatmap, row=i // 3 + 1, col=i % 3 + 1)
        fig.update_xaxes(title_text=subplot_titles[i], row=i // 3 + 1, col=i % 3 + 1)

    # Set the axis labels and title for the entire figure
    fig.update_layout(title="True - Predict", template="plotly_dark")

    # Show the figure
    fig.show()
