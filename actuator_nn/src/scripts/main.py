from ActuatorNet import ActuatorNet, train_model
import torch
import torch.nn as nn
import torch.optim as optim

# Example usage with dummy data
# Generate dummy data for demonstration purposes
def generate_dummy_data(num_samples):
    inputs = torch.randn(num_samples, 6)  # Random inputs
    labels = torch.randn(num_samples, 1)  # Random target torques
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    # Initialize the network
    net = ActuatorNet()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay for regularization

    # Generate dummy data
    train_loader = generate_dummy_data(10000)

    # Train the network
    train_model(net, criterion, optimizer, train_loader)

if __name__ == "__main__":
    main()
