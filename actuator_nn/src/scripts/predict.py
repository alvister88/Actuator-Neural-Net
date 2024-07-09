import torch
import numpy as np
import matplotlib.pyplot as plt
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
    actuals = []
    for i in range(2, num_samples):
        inputs.append([
            pos_error[i], pos_error[i-1], pos_error[i-2],
            vel[i], vel[i-1], vel[i-2]
        ])
        actuals.append(np.sin(t[i]))  # Actual torque values (sine wave)
    inputs = np.array(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    actuals = np.array(actuals)
    return inputs, actuals

def main():
    # Initialize the network
    net = ActuatorNet()

    # Load the saved weights
    load_model_weights(net, 'actuator_net_weights.pt')

    # Generate dummy test data
    test_inputs, actual_torques = generate_dummy_test_data(100)

    # Make predictions
    with torch.no_grad():  # Disable gradient computation
        predictions = net(test_inputs).numpy()

    # Print predictions
    print("Predictions:")
    print(predictions)

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(actual_torques, label='Actual Torque')
    plt.plot(predictions, label='Predicted Torque')
    plt.xlabel('Sample Index')
    plt.ylabel('Torque')
    plt.title('Predicted vs Actual Torque for Test Data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
