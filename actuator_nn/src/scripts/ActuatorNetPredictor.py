import torch
import numpy as np
from collections import deque
from ActuatorNet import ActuatorNet, INPUT_SIZE, MAX_TORQUE, MAX_VELOCITY, MAX_ERROR, MAX_ACCEL

class ActuatorNetPredictor:
    def __init__(self, model_path, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model, self.history_size = self.load_model(model_path)
        self.error_buffer = deque(maxlen=self.history_size)
        self.velocity_buffer = deque(maxlen=self.history_size)
        self.acceleration_buffer = deque(maxlen=self.history_size)

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        hidden_size = state_dict['gru.weight_ih_l0'].size(0) // 3
        num_layers = sum(1 for key in state_dict.keys() if key.startswith('gru.weight_ih_l'))
        model = ActuatorNet(hidden_size=hidden_size, num_layers=num_layers, dropout_rate=0.1)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        history_size = hidden_size
        return model, history_size

    def normalize_data(self, data, min_val, max_val):
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def denormalize_torque(self, normalized_torque):
        return (normalized_torque + 1) * (2 * MAX_TORQUE) / 2 - MAX_TORQUE

    def prepare_input(self, error, velocity, acceleration):
        # Normalize the input
        normalized_error = self.normalize_data(error, -MAX_ERROR, MAX_ERROR)
        normalized_velocity = self.normalize_data(velocity, -MAX_VELOCITY, MAX_VELOCITY)
        normalized_acceleration = self.normalize_data(acceleration, -MAX_ACCEL, MAX_ACCEL)

        self.error_buffer.append(normalized_error)
        self.velocity_buffer.append(normalized_velocity)
        self.acceleration_buffer.append(normalized_acceleration)

        # Pad the buffers if they're not full
        padded_errors = list(self.error_buffer) + [0] * (self.history_size - len(self.error_buffer))
        padded_velocities = list(self.velocity_buffer) + [0] * (self.history_size - len(self.velocity_buffer))
        padded_accelerations = list(self.acceleration_buffer) + [0] * (self.history_size - len(self.acceleration_buffer))

        input_data = np.column_stack((
            np.array(padded_errors),
            np.array(padded_velocities),
            np.array(padded_accelerations)
        ))
        return torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

    def predict(self, error, velocity, acceleration):
        input_tensor = self.prepare_input(error, velocity, acceleration)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        # Denormalize the prediction
        return self.denormalize_torque(prediction.item())

# Example usage
def main():
    model_path = '../weights/best_actuator_gru_model7.pt'  # Update this path
    predictor = ActuatorNetPredictor(model_path, device='cpu')

    # Simulating real-time data input
    sample_data = [
        (0.1, 2.0),  # (error, velocity)
        (0.2, 2.5),
        (0.15, 2.2),
        (0.18, 2.3),
        (0.22, 2.7),
    ]

    for i, (error, velocity) in enumerate(sample_data):
        prediction = predictor.predict(error, velocity)
        print(f"Sample {i+1}: Error: {error:.2f}, Velocity: {velocity:.2f}, Predicted Torque: {prediction:.4f}")

if __name__ == "__main__":
    main()