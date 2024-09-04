import torch
import numpy as np
from collections import deque
from ActuatorNet import ActuatorNet, MAX_TORQUE, MAX_VELOCITY, MAX_ERROR, MAX_ACCEL

class ActuatorNetPredictor:
    def __init__(self, model_path, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model, self.history_size, self.input_size = self.load_model(model_path)
        self.buffers = [deque(maxlen=self.history_size) for _ in range(self.input_size)]

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Determine input size, hidden size, and number of layers
        gru_weight_keys = [key for key in state_dict.keys() if key.startswith('gru.weight_ih_l')]
        num_layers = len(gru_weight_keys)
        
        if num_layers == 0:
            raise ValueError("No GRU layers found in the state dict.")
        
        hidden_size = state_dict[gru_weight_keys[0]].size(0) // 3
        input_size = state_dict[gru_weight_keys[0]].size(1)
        
        print(f"Detected model architecture: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Create the model with the detected architecture
        model = ActuatorNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=0.1)
        
        # Check if the created model matches the state dict
        model_state_dict = model.state_dict()
        if model_state_dict.keys() != state_dict.keys():
            raise RuntimeError("Mismatch between created model and loaded state dict. "
                               "Please ensure the ActuatorNet class matches the saved model architecture.")
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model, hidden_size, input_size

    def normalize_data(self, data, min_val, max_val):
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def denormalize_torque(self, normalized_torque):
        return (normalized_torque + 1) * (2 * MAX_TORQUE) / 2 - MAX_TORQUE

    def prepare_input(self, *args):
        if len(args) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, but got {len(args)}")

        # Normalize inputs
        norm_ranges = [(-MAX_ERROR, MAX_ERROR), (-MAX_VELOCITY, MAX_VELOCITY), (-MAX_ACCEL, MAX_ACCEL)]
        normalized_inputs = [self.normalize_data(arg, *norm_ranges[i]) for i, arg in enumerate(args)]

        # Add to buffers
        for buffer, value in zip(self.buffers, normalized_inputs):
            buffer.append(value)

        # Pad buffers
        padded_inputs = [list(buffer) + [0] * (self.history_size - len(buffer)) for buffer in self.buffers]

        # Create input tensor
        input_data = np.column_stack(padded_inputs)
        return torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

    def predict(self, *args):
        input_tensor = self.prepare_input(*args)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return self.denormalize_torque(prediction.item())

# Example usage
def main():
    model_path = '../weights/actuator_gruv3_model1.pt'  # Update this path
    try:
        predictor = ActuatorNetPredictor(model_path, device='cpu')
        
        print(f"Successfully loaded model with input size: {predictor.input_size}")

        # Simulating real-time data input
        sample_data = [
            (0.1, 2.0, 0.5),  # (error, velocity, acceleration)
            (0.2, 2.5, 0.7),
            (0.15, 2.2, 0.6),
            (0.18, 2.3, 0.8),
            (0.22, 2.7, 0.9),
        ]

        for i, data in enumerate(sample_data):
            prediction = predictor.predict(*data[:predictor.input_size])
            print(f"Sample {i+1}: Inputs: {data[:predictor.input_size]}, Predicted Torque: {prediction:.4f}")

    except Exception as e:
        print(f"Error loading or using the model: {str(e)}")
        print("Please ensure that the ActuatorNet class definition matches the architecture of the saved model.")

if __name__ == "__main__":
    main()