import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE, MAX_TORQUE, MAX_VELOCITY, MAX_ERROR
import time
import os

class ActuatorNetEvaluator:
    def __init__(self, model_path, run_device=None):
        self.device = self.setup_device(run_device)
        self.model = self.load_model(model_path)
        self.model_name = os.path.basename(model_path)
        self.error_values = None

    def setup_device(self, run_device):
        if run_device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(run_device)

        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        return device

    def load_model(self, model_path):
        model = ActuatorNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_data(self, file_path):
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
        position_errors = self.normalize_data(position_errors, -MAX_ERROR, MAX_ERROR)
        velocities = self.normalize_data(velocities, -MAX_VELOCITY, MAX_VELOCITY)
        torques = self.normalize_data(torques, -MAX_TORQUE, MAX_TORQUE)

        X, y = [], []
        for i in range(len(torques) - HISTORY_SIZE + 1):
            X.append(np.column_stack((position_errors[i:i+HISTORY_SIZE], 
                                      velocities[i:i+HISTORY_SIZE])))
            y.append(torques[i+HISTORY_SIZE-1])
        return np.array(X), np.array(y)

    def save_predictions(self, predictions, output_file):
        """
        Save the predicted torque values to a text file.

        Args:
            predictions (numpy.ndarray): Array of predicted torque values.
            output_file (str): Path to the output file.
        """
        np.savetxt(output_file, predictions, delimiter=',', fmt='%.6f')
        print(f"Predicted torque values saved to {output_file}")

    def evaluate_model(self, X, y, save_predictions=False, prediction_output_file=None):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor = X_tensor.view(X_tensor.size(0), -1)  # Flatten the input

        start_time = time.time()
        with torch.no_grad():
            normalized_predictions = self.model(X_tensor).cpu().numpy().flatten()
        end_time = time.time()

        predictions = self.denormalize_torque(normalized_predictions)
        y_denorm = self.denormalize_torque(y)

        self.error_values = predictions - y_denorm
        rms_error = np.sqrt(np.mean(self.error_values**2))

        total_inference_time = 1000 * (end_time - start_time)
        average_inference_time = total_inference_time / len(X) * 1000

        torque_range = np.max(y_denorm) - np.min(y_denorm)
        percentage_accuracy = (1 - (rms_error / torque_range)) * 100

        print(f'Total inference time: {total_inference_time:.4f} ms')
        print(f'Average inference time per sample: {average_inference_time:.6f} us')
        print(f'Test RMS Error: {rms_error:.3f} N·m')
        print(f'Percentage Accuracy: {percentage_accuracy:.2f}%')

        if save_predictions:
            if prediction_output_file is None:
                prediction_output_file = f'predicted_torque_{self.model_name}.txt'
            self.save_predictions(predictions, prediction_output_file)

        return predictions, y_denorm, rms_error, percentage_accuracy, total_inference_time, average_inference_time

    def plot_predictions_vs_actual(self, y, predictions, save_html=False):
        z_scores = stats.zscore(self.error_values)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.3,
                            subplot_titles=("Predictions vs Actual", "Model Error Distribution"))

        # Predictions vs Actual plot
        fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predictions vs Actual', 
                                 marker=dict(color='blue', opacity=0.7)), row=1, col=1)

        min_val, max_val = min(y.min(), predictions.min()), max(y.max(), predictions.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                                 name='Ideal Line', line=dict(dash='dash', color='darkred')), row=1, col=1)

        # Error histogram
        fig.add_trace(go.Histogram(x=self.error_values, name='Error Distribution', histnorm='probability', 
                                   marker_color='rgba(64, 224, 208, 0.8)'), row=2, col=1)

        # Update axes
        fig.update_xaxes(title_text='Actual Torque (N·m)', row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='Predicted Torque (N·m)', row=1, col=1, tickformat=".2f", showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text='Model Error (N·m)', row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='Probability', row=2, col=1, showgrid=True, gridcolor='lightgray', dtick=0.005)

        # Add secondary x-axis for z-scores
        fig.update_xaxes(title_text='Z-Score', overlaying='x', side='top', row=2, col=1,
                         range=[z_scores.min(), z_scores.max()], showgrid=False)

        # Add vertical lines for standard deviations
        for sd in [-3, -2, -1, 0, 1, 2, 3]:
            fig.add_vline(x=np.mean(self.error_values) + sd * np.std(self.error_values), line_dash="dash", 
                          line_color="red", annotation_text=f"{sd}σ", annotation_position="bottom", row=2, col=1)

        # Error statistics
        rms_error = np.sqrt(np.mean(self.error_values**2))
        mean_error = np.mean(self.error_values)
        error_variance = np.var(self.error_values)

        # Annotations
        fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.15,
                           text=f"RMS Error: {rms_error:.3f} N·m<br>Mean Error: {mean_error:.3f} N·m<br>Error Variance: {error_variance:.6f}",
                           showarrow=False, font=dict(size=12), align="right",
                           bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1)

        # Update layout
        fig.update_layout(height=800, title_text=f'Actuator Network Predictions vs Actual Torque - Model: {self.model_name}',
                          showlegend=True, plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
                          yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True))

        fig.show()

        if save_html:
            fig.write_html(f'predictions_vs_actual_{self.model_name}.html')

    def plot_data_visualization(self, y, predictions, position_errors, velocities, rms_error, 
                                percentage_accuracy, total_inference_time, average_inference_time, 
                                plot_vs_time=False, save_html=False):
        if plot_vs_time and hasattr(self, 'time_values'):
            x_axis = self.time_values[-len(y):] - self.time_values[-len(y)]
            x_axis_label = 'Time [s]'
        else:
            x_axis = np.arange(len(y))
            x_axis_label = 'Sample'

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        # Position Error plot
        fig.add_trace(go.Scatter(x=x_axis, y=position_errors[-len(y):], mode='lines', name='Position Error', 
                                 line=dict(color='purple', width=3)), row=1, col=1)

        # Velocity plot
        fig.add_trace(go.Scatter(x=x_axis, y=velocities[-len(y):], mode='lines', name='Velocity', 
                                 line=dict(color='blue', width=3)), row=2, col=1)

        # Predicted vs Actual Torque plot
        fig.add_trace(go.Scatter(x=x_axis, y=predictions, mode='lines', name='Predicted Torque', 
                                 line=dict(color='#009FBD', width=3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=y, mode='lines', name='Actual Torque', 
                                 line=dict(color='#161D6F', width=3, dash='10px, 2px')), row=3, col=1)

        # Model Error plot
        fig.add_trace(go.Scatter(x=x_axis, y=self.error_values, mode='lines', name='Prediction Error', 
                                 line=dict(color='red', width=3)), row=4, col=1)

        # Update axes
        y_titles = ['Position Error [rad]', 'Velocity [units/s]', 'Torque [Nm]', 'Model Error [Nm]']
        for i, title in enumerate(y_titles, start=1):
            fig.update_yaxes(title_text=title, row=i, col=1, showline=True, linecolor='lightgray', 
                             linewidth=3, ticks='inside', tickcolor='lightgray', ticklen=8, 
                             tickwidth=3, showgrid=False, mirror='ticks')

        for i in range(1, 5):
            fig.update_xaxes(title_text=x_axis_label, row=i, col=1, showline=True, linecolor='lightgray', 
                             linewidth=3, ticks='inside', tickcolor='lightgray', ticklen=8, 
                             tickwidth=3, showgrid=False, mirror='ticks', showticklabels=True)

        # Update layout
        fig.update_layout(height=1600, title_text=f'Data Visualization - Model: {self.model_name}',
                          showlegend=True, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=50, r=50, t=50, b=50))

        fig.show()

        if save_html:
            fig.write_html(f'plot_data_visualization_{self.model_name}.html')

def main():
    data_path = 'path/to/your/data.csv'
    model_path = 'path/to/your/model.pt'

    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    
    position_errors, velocities, torques = evaluator.load_data(data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)
    
    predictions, y_denorm, rms_error, percentage_accuracy, total_inference_time, average_inference_time = evaluator.evaluate_model(X, y)
    
    evaluator.plot_predictions_vs_actual(y_denorm, predictions, save_html=True)
    evaluator.plot_data_visualization(y_denorm, predictions, position_errors, velocities, rms_error, 
                                      percentage_accuracy, total_inference_time, average_inference_time, 
                                      plot_vs_time=False, save_html=True)

if __name__ == "__main__":
    main()