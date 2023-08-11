import plotly.graph_objs as go
from plotly.subplots import make_subplots


def show_digits(digits, labels, wrong):

    # Create a 2x3 subplot for the digits
    fig = make_subplots(rows=2, cols=3)

    # Add each digit to the subplot as a heatmap, along with its label
    for i in range(6):
        digit = digits[i][::-1]
        heatmap = go.Heatmap(z=digit, colorscale='gray')
        fig.add_trace(heatmap, row=i // 3 + 1, col=i % 3 + 1)
        fig.add_annotation(
            dict(text=f"{labels[i]} - {wrong[i]}", xref="x" + str(i + 1), yref="y" + str(i + 1), showarrow=False,
                 font=dict(color="white")))

    # Set the axis labels and title
    fig.update_layout(xaxis=dict(title='Column'), yaxis=dict(title='Row'), title='Digits Heatmap')

    # Show the figure
    fig.show()
