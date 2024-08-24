import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ActuatorNet import ActuatorNet  # Ensure this import works with your file structure
from ActuatorNetTrainer import ActuatorNetTrainer

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    position_errors = data['Error'].values
    velocities = data['Velocity'].values
    torques = data['Torque'].values
    return position_errors, velocities, torques

def prepare_sequence_data(position_errors, velocities, torques, sequence_length=3):
    X, y = [], []
    for i in range(len(torques) - sequence_length + 1):
        X.append(np.column_stack((position_errors[i:i+sequence_length], 
                                  velocities[i:i+sequence_length])).flatten())
        y.append(torques[i+sequence_length-1])
    return np.array(X), np.array(y)

def load_model(model_path, dropout_rate=0.2):
    model = ActuatorNet(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, X, y, position_errors, velocities, torques):
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X)).numpy().flatten()

    # Calculate RMS error
    rms_error = np.sqrt(np.mean((predictions - y) ** 2))
    print(f'Test RMS Error: {rms_error:.3f} N·m')

    # Calculate the range of torque values
    torque_range = np.max(y) - np.min(y)
    
    # Calculate percentage accuracy
    percentage_accuracy = (1 - (rms_error / torque_range)) * 100
    print(f'Percentage Accuracy: {percentage_accuracy:.2f}%')

    # Plot predictions vs actual using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predictions vs Actual'))
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ideal Line', line=dict(dash='dash')))
    fig.update_layout(
        title='Actuator Network Predictions vs Actual Torque',
        xaxis_title='Actual Torque (N·m)',
        yaxis_title='Predicted Torque (N·m)',
        yaxis=dict(tickformat=".2f")  # Format the y-axis to show more decimal places for clarity
    )

    # Adding horizontal lines at specific intervals
    for i in range(int(np.min(predictions)), int(np.max(predictions)) + 1, 2):  # Adjust the interval as needed
        fig.add_hline(y=i, line_dash="dash", line_color="gray")

    fig.show()

    # Plot error, velocity, predicted torque, and actual torque using Plotly
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01)

    fig.add_trace(go.Scatter(y=position_errors[-len(y):], mode='lines', name='Error'), row=1, col=1)
    fig.update_yaxes(title_text='Position Error (rad)', row=1, col=1, tickformat=".2f", dtick=0.5)  # Specify units if known

    fig.add_trace(go.Scatter(y=velocities[-len(y):], mode='lines', name='Velocity'), row=2, col=1)
    fig.update_yaxes(title_text='Velocity (units/s)', row=2, col=1, tickformat=".2f", dtick=5.0)  # Specify units if known

    fig.add_trace(go.Scatter(y=y, mode='lines', name='Actual Torque', line=dict(dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted Torque'), row=3, col=1)
    fig.update_yaxes(title_text='Torque (N·m)', row=3, col=1, tickformat=".2f", dtick=0.25)

    fig.add_trace(go.Scatter(y=y - predictions, mode='lines', name='Prediction Error'), row=4, col=1)
    fig.update_yaxes(title_text='Model Error (N·m)', row=4, col=1, tickformat=".2f", dtick=0.1)

    # Adding horizontal lines to the subplots
    for i in range(int(np.min(y - predictions)), int(np.max(y - predictions)) + 1, 2):  # Adjust the interval as needed
        fig.add_hline(y=i, line_dash="dash", line_color="gray", row=4, col=1)

    fig.update_layout(height=800, title_text='Data Visualization', showlegend=True)
    fig.show()

    return predictions

def main():
    # Load the dataset
    data_path = '../data/contact1.txt'  # Update this path as needed
    position_errors, velocities, torques = load_data(data_path)

    # Prepare the data
    X, y = prepare_sequence_data(position_errors, velocities, torques)

    # Load the trained model
    model_path = '../weights/best_actuator_model6.pt'  # Update this path as needed
    model = load_model(model_path)
    
    # Evaluate the model
    evaluate_model(model, X, y, position_errors, velocities, torques)

if __name__ == "__main__":
    main()
