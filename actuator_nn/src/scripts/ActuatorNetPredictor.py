import torch
import numpy as np
from collections import deque
from ActuatorNet import ActuatorNet, HISTORY_SIZE  # Make sure this import works with your project structure

class ActuatorNetPredictor:
    def __init__(self, model_path, sequence_length=HISTORY_SIZE, device=None):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = self.load_model(model_path)
        self.error_buffer = deque(maxlen=sequence_length)
        self.velocity_buffer = deque(maxlen=sequence_length)

    def load_model(self, model_path):
        model = ActuatorNet(dropout_rate=0.1)  # Adjust dropout_rate if needed
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def prepare_input(self, error, velocity):
        self.error_buffer.append(error)
        self.velocity_buffer.append(velocity)
        
        if len(self.error_buffer) < self.sequence_length:
            return None  # Not enough data yet
        
        input_data = np.concatenate([
            np.array(self.error_buffer),
            np.array(self.velocity_buffer)
        ]).flatten()
        
        return torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

    def predict(self, error, velocity):
        input_tensor = self.prepare_input(error, velocity)
        if input_tensor is None:
            return None  # Not enough data for prediction yet
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        return prediction.item()

# Example usage
def main():
    model_path = '../weights/best_actuator_model12.pt'  # Update this path
    predictor = ActuatorNetPredictor(model_path, sequence_length=3)

    # Simulating real-time data input
    sample_data = [
        (0.1, 2.0),  # (error, velocity)
        (0.2, 2.5),
        (0.15, 2.2),
        (0.18, 2.3),
        (0.22, 2.7),
    ]

    for error, velocity in sample_data:
        prediction = predictor.predict(error, velocity)
        if prediction is not None:
            print(f"Error: {error:.2f}, Velocity: {velocity:.2f}, Predicted Torque: {prediction:.4f}")
        else:
            print(f"Error: {error:.2f}, Velocity: {velocity:.2f}, Accumulating data...")

if __name__ == "__main__":
    main()