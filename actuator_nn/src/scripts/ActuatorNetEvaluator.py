import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE, NUM_LAYERS, MAX_TORQUE, MAX_VELOCITY, MAX_ERROR
import time
import os

class ActuatorNetEvaluator:
    def __init__(self, model_path, run_device=None, input_size=2):
        self.model, self.device, self.hidden_size, self.num_layers = self.load_model(model_path, run_device)
        self.input_size = input_size
        self.history_size = self.hidden_size  # Assuming HISTORY_SIZE is the same as hidden_size
        self.model_name = os.path.basename(model_path)
        self.file_path = None

        # for sharing between graphs
        self.error_variance = None
        self.error_values = None

    def load_data(self, file_path):
        self.file_path = file_path
        data = pd.read_csv(file_path, delimiter=',')
        position_errors = data['Error'].values
        velocities = data['Velocity'].values
        torques = data['Torque'].values
        return position_errors, velocities, torques

    def normalize_data(self, data, min_val, max_val):
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def denormalize_torque(self, normalized_torque):
        return (normalized_torque + 1) * (2 * MAX_TORQUE) / 2 - MAX_TORQUE

    def prepare_sequence_data(self, position_errors, velocities, torques):
        # Normalize only the input data
        position_errors = self.normalize_data(position_errors, -MAX_ERROR, MAX_ERROR)
        velocities = self.normalize_data(velocities, -MAX_VELOCITY, MAX_VELOCITY)

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
            normalized_predictions = self.model(X_tensor).cpu().numpy().flatten()
            predictions = self.denormalize_torque(normalized_predictions)
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

        self.plot_predictions_vs_actual(y, predictions) # This graph has to go first since next graph relies on some of its calculations
        self.plot_data_visualization(y, predictions, position_errors, velocities, rms_error, percentage_accuracy, total_inference_time, average_inference_time)

        return {
            'total_inference_time': total_inference_time,
            'average_inference_time': average_inference_time,
            'rms_error': rms_error,
            'percentage_accuracy': percentage_accuracy
        }

    def plot_predictions_vs_actual(self, y, predictions):
        # Calculate model error
        self.error_values = predictions - y

        # Calculate z-scores
        z_scores = stats.zscore(self.error_values)

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.3,
                            subplot_titles=("Predictions vs Actual", "Model Error Distribution"))

        # Predictions vs Actual plot
        fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predictions vs Actual'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ideal Line', line=dict(dash='dash')), row=1, col=1)

        # Error histogram with z-scores
        fig.add_trace(go.Histogram(x=self.error_values, name='Error Distribution', histnorm='probability'), row=2, col=1)

        # Update layout for each subplot
        fig.update_xaxes(title_text='Actual Torque (N·m)', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Torque (N·m)', tickformat=".2f", row=1, col=1)
        fig.update_xaxes(title_text='Model Error (N·m)', row=2, col=1)
        fig.update_yaxes(title_text='Probability', row=2, col=1, dtick=0.005)

        # Add secondary x-axis for z-scores
        fig.update_xaxes(
            title_text='Z-Score',
            overlaying='x',
            side='top',
            row=2, col=1,
            range=[z_scores.min(), z_scores.max()],
            showgrid=False
        )

        # Add vertical lines for standard deviations
        for sd in [-3, -2, -1, 0, 1, 2, 3]:
            fig.add_vline(x=np.mean(self.error_values) + sd * np.std(self.error_values), line_dash="dash", line_color="red",
                          annotation_text=f"{sd}σ", annotation_position="bottom", row=2, col=1)

        # Add horizontal lines to Predictions vs Actual plot
        for i in range(int(np.min(predictions)), int(np.max(predictions)) + 1, 2):
            fig.add_shape(type="line", x0=y.min(), x1=y.max(), y0=i, y1=i,
                          line=dict(dash="dash", color="gray"), row=1, col=1)

        # Calculate error statistics
        rms_error = np.sqrt(np.mean(self.error_values**2))
        mean_error = np.mean(self.error_values)
        self.error_variance = np.var(self.error_values)

        # Add annotations for error statistics
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.15,
            text=f"RMS Error: {rms_error:.3f} N·m<br>Mean Error: {mean_error:.3f} N·m<br>Error Variance: {self.error_variance:.6f}",
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )

        # Update overall layout
        fig.update_layout(
            height=800,
            title_text=f'Actuator Network Predictions vs Actual Torque - Model: {self.model_name}',
            showlegend=True
        )

        fig.show()

    def plot_data_visualization(self, y, predictions, position_errors, velocities, rms_error, percentage_accuracy, total_inference_time, average_inference_time):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01)

        fig.add_trace(go.Scatter(y=position_errors[-len(y):], mode='lines', name='Error'), row=1, col=1)
        fig.update_yaxes(title_text='Position Error (rad)', row=1, col=1, tickformat=".2f", dtick=0.5)

        fig.add_trace(go.Scatter(y=velocities[-len(y):], mode='lines', name='Velocity'), row=2, col=1)
        fig.update_yaxes(title_text='Velocity (units/s)', row=2, col=1, tickformat=".2f", dtick=5.0)

        # Calculate model variance +-
        std_dev = np.sqrt(self.error_variance)

        # Create arrays for upper and lower bounds
        upper_bound = predictions + 2 * std_dev
        lower_bound = predictions - 2 * std_dev

        # Add actual torque
        fig.add_trace(go.Scatter(y=y, mode='lines', name='Actual Torque', line=dict(color='#17becf', dash='dot')), row=3, col=1)

        # Add predicted torque with variance
        fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted Torque', line=dict(color='#B967FF')), row=3, col=1)
        fig.add_trace(go.Scatter(
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            name='Variance',
            hoverinfo='skip'
        ), row=3, col=1)

        fig.update_yaxes(title_text='Torque (N·m)', row=3, col=1, tickformat=".2f", dtick=5.0)

        fig.add_trace(go.Scatter(y=self.error_values, mode='lines', name='Prediction Error'), row=4, col=1)
        fig.update_yaxes(title_text='Model Error (N·m)', row=4, col=1, tickformat=".2f", dtick=1.0)

        for i in range(int(np.min(self.error_values)), int(np.max(self.error_values)) + 1, 2):
            fig.add_hline(y=i, line_dash="dash", line_color="gray", row=4, col=1)

        annotations = [
            f"Model: {self.model_name}",
            f"File Path: {self.file_path}",
            f"Device: {self.device.type}",
            f"Test RMS Error: {rms_error:.3f} N·m",
            f"Accuracy: {percentage_accuracy:.2f}%",
            f"Total inference: {total_inference_time:.4f} ms",
            f"Avg. inference: {average_inference_time:.6f} us",
            f"Error Variance: {self.error_variance:.6f}"
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