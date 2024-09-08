import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE, NUM_LAYERS, MAX_TORQUE, MAX_VELOCITY, MAX_ERROR, MAX_ACCEL
import time
import os

class ActuatorNetEvaluator:
    def __init__(self, model_path, run_device=None, input_size=3):
        self.model, self.device, self.hidden_size, self.num_layers = self.load_model(model_path, run_device)
        self.input_size = input_size
        self.history_size = self.hidden_size  # Assuming HISTORY_SIZE is the same as hidden_size
        self.model_name = os.path.basename(model_path)

        # for sharing between graphs
        self.error_variance = None
        self.error_values = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path, delimiter=',')
        position_errors = data['Error'].values
        velocities = data['Velocity'].values
        accelerations = data['Acceleration'].values  
        torques = data['Torque'].values

        # Store time values if available
        if 'Time' in data.columns:
            self.time_values = data['Time'].values

        return position_errors, velocities, accelerations, torques   

    def normalize_data(self, data, min_val, max_val):
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def denormalize_torque(self, normalized_torque):
        return (normalized_torque + 1) * (2 * MAX_TORQUE) / 2 - MAX_TORQUE

    def prepare_sequence_data(self, position_errors, velocities, accelerations, torques):
        # Normalize the input data
        position_errors = self.normalize_data(position_errors, -MAX_ERROR, MAX_ERROR)
        velocities = self.normalize_data(velocities, -MAX_VELOCITY, MAX_VELOCITY)
        accelerations = self.normalize_data(accelerations, -MAX_ACCEL, MAX_ACCEL) 

        X, y = [], []
        for i in range(len(torques) - self.history_size + 1):
            X.append(np.column_stack((position_errors[i:i+self.history_size], 
                                      velocities[i:i+self.history_size],
                                      accelerations[i:i+self.history_size])))  
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
        self.input_size = state_dict['gru.weight_ih_l0'].size(1)
        num_layers = sum(1 for key in state_dict.keys() if key.startswith('gru.weight_ih_l'))
        
        # Create the model with the correct architecture
        model = ActuatorNet(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=0.1)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
                
        return model, device, hidden_size, num_layers

    def evaluate_model(self, X, y, position_errors, velocities, accelerations, torques, vs_time=False, save_html=False):
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

        self.plot_predictions_vs_actual(y, predictions)
        self.plot_data_visualization(y, predictions, position_errors, velocities, accelerations, 
                                     rms_error, percentage_accuracy, total_inference_time, average_inference_time, 
                                     vs_time, save_html)

        return {
            'total_inference_time': total_inference_time,
            'average_inference_time': average_inference_time,
            'rms_error': rms_error,
            'percentage_accuracy': percentage_accuracy
        }

    def plot_predictions_vs_actual(self, y, predictions, save_html=False):
        # Calculate model error
        self.error_values = predictions - y
        z_scores = stats.zscore(self.error_values)

        # Create subplots with shared layout adjustments
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.3,
            subplot_titles=("Predictions vs Actual", "Model Error Distribution")
        )

        # Predictions vs Actual plot
        fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predictions vs Actual', marker=dict(color='blue', opacity=0.7)), row=1, col=1)
        
        # Ensure the ideal line spans the range of actual values on the x-axis (Actual Torque)
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())

        # Plot the ideal line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],  # Ensure the x-range matches the full range of values
            y=[min_val, max_val],  # Ensure the y-range is the same for a diagonal line
            mode='lines',
            name='Ideal Line',
            line=dict(dash='dash', color='darkred')
        ), row=1, col=1)


        # Error histogram with z-scores and custom color for the bars
        fig.add_trace(go.Histogram(
            x=self.error_values, 
            name='Error Distribution', 
            histnorm='probability', 
            marker_color='rgba(64, 224, 208, 0.8)'  # light greenish turquoise
        ), row=2, col=1)

        # Update x/y axes and grid lines for MATLAB-like appearance
        fig.update_xaxes(title_text='Actual Torque (N·m)', row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='Predicted Torque (N·m)', row=1, col=1, tickformat=".2f", showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text='Model Error (N·m)', row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='Probability', row=2, col=1, showgrid=True, gridcolor='lightgray', dtick=0.005)

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

        # Horizontal grid lines for Predictions vs Actual plot
        # for i in range(int(np.min(predictions)), int(np.max(predictions)) + 1, 2):
        #     fig.add_shape(type="line", x0=y.min(), x1=y.max(), y0=i, y1=i,
        #                 line=dict(dash="dash", color="gray"), row=1, col=1)

        # Error statistics
        rms_error = np.sqrt(np.mean(self.error_values**2))
        mean_error = np.mean(self.error_values)
        self.error_variance = np.var(self.error_values)

        # Annotations
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

        # Update overall layout for MATLAB style
        fig.update_layout(
            height=800,
            title_text=f'Actuator Network Predictions vs Actual Torque - Model: {self.model_name}',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        )

        fig.show()
        
        # Save the plot as HTML if specified
        if save_html:
            fig.write_html(f'predictions_vs_actual_{self.model_name}.html')


    def plot_data_visualization(self, y, predictions, position_errors, velocities, accelerations, 
                            rms_error, percentage_accuracy, total_inference_time, average_inference_time, 
                            plot_vs_time=False, save_html=False):
        """
        Plot data visualization either versus time or sample indices, with a light gray border around each subplot.
        
        Parameters:
            plot_vs_time (bool): If True, plot versus the 'Time' column. Otherwise, plot versus sample indices.
        """
        sampling_rate = 300  # Sampling rate in Hz (if plotting by samples)

        # Choose x-axis: either time values from the file or sample indices
        if plot_vs_time and self.time_values is not None:
            x_axis = self.time_values[-len(y):]  # Time values (truncate to match length of y)
            x_axis_label = 'Time [s]'
        else:
            x_axis = np.arange(len(y)) / sampling_rate  # Sample indices (converted to time in seconds)
            x_axis_label = 'Samples' if not plot_vs_time else 'Time [s]'

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        # Position Error plot
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=position_errors[-len(y):], 
            mode='lines', 
            name='Position Error', 
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        fig.update_yaxes(title_text='Position Error [rad]', row=1, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror for all sides
        fig.update_xaxes(showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, row=1, col=1, mirror=True)  # Add mirror

        # Velocity plot
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=velocities[-len(y):], 
            mode='lines', 
            name='Velocity', 
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        fig.update_yaxes(title_text='Velocity [units/s]', row=2, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror for all sides
        fig.update_xaxes(showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, row=2, col=1, mirror=True)  # Add mirror

        # Acceleration plot
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=accelerations[-len(y):], 
            mode='lines', 
            name='Acceleration', 
            line=dict(color='green', width=2)
        ), row=3, col=1)
        fig.update_yaxes(title_text='Acceleration [units/s²]', row=3, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror for all sides
        fig.update_xaxes(showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, row=3, col=1, mirror=True)  # Add mirror

        # Predicted vs Actual Torque plot
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=y, 
            mode='lines', 
            name='Actual Torque', 
            line=dict(color='black', width=2, dash='dot')
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=predictions, 
            mode='lines', 
            name='Predicted Torque', 
            line=dict(color='red', width=2)
        ), row=4, col=1)

        fig.update_yaxes(title_text='Torque [Nm]', row=4, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror for all sides
        fig.update_xaxes(showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, row=4, col=1, mirror=True)  # Add mirror

        # Model Error plot
        fig.add_trace(go.Scatter(
            x=x_axis, 
            y=self.error_values, 
            mode='lines', 
            name='Prediction Error', 
            line=dict(color='red', width=2)
        ), row=5, col=1)
        fig.update_yaxes(title_text='Model Error [Nm]', row=5, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror for all sides
        fig.update_xaxes(title_text=x_axis_label, row=5, col=1, 
                        showline=True, linecolor='lightgray', linewidth=2,  # Add border
                        ticks='inside', tickcolor='black', showgrid=False, mirror=True)  # Add mirror

        # Update layout for a clean style with a light gray border around each subplot
        fig.update_layout(
            height=1800,  # Adjust height based on your preference
            title_text=f'Data Visualization - Model: {self.model_name}',
            showlegend=True,
            plot_bgcolor='white',  # White background for a clean look
            paper_bgcolor='white',  # White background for the figure
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins to remove clutter
        )

        fig.show()
        
        # Save the plot as HTML if specified
        if save_html:
            fig.write_html(f'plot_data_visualization_{self.model_name}.html')




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