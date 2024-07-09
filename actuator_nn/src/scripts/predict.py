import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ActuatorNet import ActuatorNet

# Load the model weights
def load_model_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()  # Set to evaluation mode
    print("Model weights loaded.")

# Generate dummy data for testing predictions
def generate_dummy_test_data(num_samples):
    t = np.linspace(0, 2 * np.pi, num_samples)
    pos_error = np.sin(t) + 0.1 * np.random.randn(num_samples)
    vel = np.cos(t) + 0.1 * np.random.randn(num_samples)
    inputs = []
    for i in range(2, num_samples):
        inputs.append([
            pos_error[i], pos_error[i-1], pos_error[i-2],
            vel[i], vel[i-1], vel[i-2]
        ])
    inputs = np.array(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def main():
    # Initialize the network
    net = ActuatorNet()

    # Load the saved weights
    load_model_weights(net, 'actuator_net_weights.pt')

    # Generate dummy test data
    test_inputs = generate_dummy_test_data(100)

    # Make predictions
    with torch.no_grad():  # Disable gradient computation
        predictions = net(test_inputs)

    # Print predictions
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
