from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Example usage with dummy data
# Generate dummy data for demonstration purposes
def generate_dummy_data(num_samples, batch_size=32):
    inputs = torch.randn(num_samples, 6)  # Random inputs
    labels = torch.randn(num_samples, 1)  # Random target torques
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generate dummy data for demonstration purposes
def generate_dummy_sine_data(num_samples, batch_size=32):
    # Time steps
    t = np.linspace(0, 2 * np.pi, num_samples)
    
    # Sine wave for position errors
    pos_error = np.sin(t)
    
    # Velocity is the derivative of the position
    vel = np.cos(t)
    
    # Add Gaussian noise to position errors and velocities
    noise_level = 0.1
    pos_error += noise_level * np.random.randn(num_samples)
    vel += noise_level * np.random.randn(num_samples)
    
    # Create input features: current state and past two states
    inputs = []
    for i in range(2, num_samples):
        inputs.append([
            pos_error[i], pos_error[i-1], pos_error[i-2],
            vel[i], vel[i-1], vel[i-2]
        ])
    
    inputs = np.array(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    
    # Target torque (dummy values, can be replaced with actual computation)
    labels = np.sin(t[2:]).reshape(-1, 1)  # Just as an example, use sine wave as target
    labels = torch.tensor(labels, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    # Initialize the network
    net = ActuatorNet()
    trainer = ActuatorNetTrainer()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay for regularization

    # Generate dummy data
    train_loader = generate_dummy_sine_data(10000)
    val_loader = generate_dummy_sine_data(2000)

    # Train the network
    trainer.train_model(net, criterion, optimizer, train_loader, val_loader, num_epochs=30)

if __name__ == "__main__":
    main()
