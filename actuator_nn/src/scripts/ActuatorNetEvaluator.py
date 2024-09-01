import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE, NUM_LAYERS
import time
import os

class ActuatorNetEvaluator:
    def __init__(self, model_path, run_device=None, input_size=2):
        self.model, self.device, self.hidden_size, self.num_layers = self.load_model(model_path, run_device)
        self.input_size = input_size
        self.history_size = self.hidden_size  # Assuming HISTORY_SIZE is the same as hidden_size
        self.model_name = os.path.basename(model_path)

    def load_data(self, file_path):
        data = pd.read_csv(file_path, delimiter=',')
        position_errors = data['Error'].values
        velocities = data['Velocity'].values
        torques = data['Torque'].values
        return position_errors, velocities, torques

    def prepare_sequence_data(self, position_errors, velocities, torques):
        X, y = [], []
        for i in range(len(torques) - self.history_size + 1):
            X.append(np.column_stack((position_errors[i:i+self.history_size], 
                                      velocities[i:i+self.history_size])))
            y.append(torques[i+self.history_size-1])
        return np.array(X), np.array(y)

    def load_model(self, model_path, run_device=None):
        if run_device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(run_device)

        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Determine the hidden size and number of layers from the state dict
        hidden_size = state_dict['gru.weight_ih_l0'].size(0) // 3
        num_layers = sum(1 for key in state_dict.keys() if key.startswith('gru.weight_ih_l'))
        
        # Create the model with the correct architecture
        model = ActuatorNet(hidden_size=hidden_size, num_layers=num_layers, dropout_rate=0.1)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
                
        return model, device, hidden_size, num_layers

    def evaluate_model(self, X, y, position_errors, velocities, torques):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        end_time = time.time()

        total_inference_time = 1000 * (end_time - start_time) 
        average_inference_time = 1000 * total_inference_time / len(X)

        print(f'Total inference time: {total_inference_time:.4f} ms')
        print(f'Average inference time per sample: {average_inference_time:.6f} us')

        rms_error = np.sqrt(np.mean((predictions - y) ** 2))
        print(f'Test RMS Error: {rms_error:.3f} N·m')

        torque_range = np.max(y) - np.min(y)
        percentage_accuracy = (1 - (rms_error / torque_range)) * 100
        print(f'Percentage Accuracy: {percentage_accuracy:.2f}%')

        self.plot_predictions_vs_actual(y, predictions)
        self.plot_data_visualization(y, predictions, position_errors, velocities, rms_error, percentage_accuracy, total_inference_time, average_inference_time)

        return {
            'total_inference_time': total_inference_time,
            'average_inference_time': average_inference_time,
            'rms_error': rms_error,
            'percentage_accuracy': percentage_accuracy
        }

    def plot_predictions_vs_actual(self, y, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predictions vs Actual'))
        fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ideal Line', line=dict(dash='dash')))
        fig.update_layout(
            title=f'Actuator Network Predictions vs Actual Torque - Model: {self.model_name}',
            xaxis_title='Actual Torque (N·m)',
            yaxis_title='Predicted Torque (N·m)',
            yaxis=dict(tickformat=".2f")
        )

        for i in range(int(np.min(predictions)), int(np.max(predictions)) + 1, 2):
            fig.add_hline(y=i, line_dash="dash", line_color="gray")

        fig.show()

    def plot_data_visualization(self, y, predictions, position_errors, velocities, rms_error, percentage_accuracy, total_inference_time, average_inference_time):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01)

        fig.add_trace(go.Scatter(y=position_errors[-len(y):], mode='lines', name='Error'), row=1, col=1)
        fig.update_yaxes(title_text='Position Error (rad)', row=1, col=1, tickformat=".2f", dtick=0.5)

        fig.add_trace(go.Scatter(y=velocities[-len(y):], mode='lines', name='Velocity'), row=2, col=1)
        fig.update_yaxes(title_text='Velocity (units/s)', row=2, col=1, tickformat=".2f", dtick=5.0)

        fig.add_trace(go.Scatter(y=y, mode='lines', name='Actual Torque', line=dict(dash='dot')), row=3, col=1)
        fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted Torque'), row=3, col=1)
        fig.update_yaxes(title_text='Torque (N·m)', row=3, col=1, tickformat=".2f", dtick=0.25)

        fig.add_trace(go.Scatter(y=y - predictions, mode='lines', name='Prediction Error'), row=4, col=1)
        fig.update_yaxes(title_text='Model Error (N·m)', row=4, col=1, tickformat=".2f", dtick=0.05)

        for i in range(int(np.min(y - predictions)), int(np.max(y - predictions)) + 1, 2):
            fig.add_hline(y=i, line_dash="dash", line_color="gray", row=4, col=1)

        annotations = [
            f"Model: {self.model_name}",
            f"Device: {self.device.type}",
            f"Test RMS Error: {rms_error:.3f} N·m",
            f"Accuracy: {percentage_accuracy:.2f}%",
            f"Total inference: {total_inference_time:.4f} ms",
            f"Avg. inference: {average_inference_time:.6f} us",
        ]

        for i, annotation in enumerate(annotations):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.11, y=0.48 - i*0.04,
                text=annotation,
                showarrow=False,
                font=dict(size=11),
                align="left",
            )

        fig.update_layout(height=1200, title_text=f'Data Visualization - Model: {self.model_name}', showlegend=True)
        fig.show()

def main():
    # Update these paths as needed
    data_path = '../data/normal1.txt'
    model_path = '../weights/best_actuator_model_gru.pt'

    # Create an instance of the evaluator
    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')

    # Load and prepare the data
    position_errors, velocities, torques = evaluator.load_data(data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)

    # Evaluate the model
    metrics = evaluator.evaluate_model(X, y, position_errors, velocities, torques)

    # Print the evaluation metrics
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()