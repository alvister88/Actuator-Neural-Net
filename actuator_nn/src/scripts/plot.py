import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_data(file_path):
    return pd.read_csv(file_path, delimiter=',')

def plot_data(data):
    # Create a figure with subplots
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('Position', 'Velocity', 'Acceleration', 'Error', 'Current', 'Torque'))

    # Create a sample index as a list
    samples = list(range(len(data)))

    # Plot Samples vs TargetPosition and CurrentPosition
    fig.add_trace(go.Scatter(x=samples, y=data['TargetPosition'], name='Target Position'), row=1, col=1)
    fig.add_trace(go.Scatter(x=samples, y=data['CurrentPosition'], name='Current Position'), row=1, col=1)

    # Plot Samples vs Velocity
    fig.add_trace(go.Scatter(x=samples, y=data['Velocity'], name='Velocity'), row=2, col=1)

    # Plot Samples vs Acceleration
    fig.add_trace(go.Scatter(x=samples, y=data['Acceleration'], name='Acceleration'), row=3, col=1)

    # Plot Samples vs Error
    fig.add_trace(go.Scatter(x=samples, y=data['Error'], name='Error'), row=4, col=1)

    # Plot Samples vs Current
    fig.add_trace(go.Scatter(x=samples, y=data['Current'], name='Current', line=dict(color='pink')), row=5, col=1)

    # Plot Samples vs Torque
    fig.add_trace(go.Scatter(x=samples, y=data['Torque'], name='Torque', line=dict(color='cyan')), row=6, col=1)


    # Update layout
    fig.update_layout(height=1400, width=1700, title_text="Actuator Data Visualization")
    fig.update_xaxes(title_text="Samples", row=6, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration", row=3, col=1)
    fig.update_yaxes(title_text="Error", row=4, col=1)
    fig.update_yaxes(title_text="Current", row=5, col=1)
    fig.update_yaxes(title_text="Torque", row=6, col=1)

    # Show the plot
    fig.show()

def main():
    while True:
        # Ask for file name
        file_name = input("Enter the name of the file to load (or 'q' to quit): ")
        if file_name.lower() == 'q':
            print("Exiting the program.")
            break

        # Construct the full file path
        file_path = os.path.join('..', 'data', file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_name}' does not exist in the specified directory.")
            continue

        try:
            # Load the data
            data = load_data(file_path)

            # Check if 'Current' column exists
            if 'Current' not in data.columns:
                print("Warning: 'Current' column not found in the data. Skipping current plot.")
                continue

            # Plot the data
            plot_data(data)
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()