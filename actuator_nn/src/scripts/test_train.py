from actuator_nn.src.utils.DCMotorPlant import DCMotorPlant
from actuator_nn.src.scripts.ActuatorNet import ActuatorNet
from actuator_nn.src.scripts.ActuatorNetTrainer import ActuatorNetTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Generate training data using the DC motor plant simulation
def generate_motor_plant_data(num_samples, noise_levels):
    motor = DCMotorPlant()
    motor.set_noise(**noise_levels)

    # Time vector
    t = np.linspace(0, 10, num_samples)
    dt = t[1] - t[0]

    # Desired trajectory (sine wave for position)
    position_desired = np.sin(t)

    # Simulate the motor
    i = np.zeros_like(t)
    omega = np.zeros_like(t)
    V = np.zeros_like(t)
    position_actual = np.zeros_like(t)
    y = [0.0, 0.0]  # Initial conditions: [current, angular velocity]

    inputs = []
    labels = []

    for idx in range(1, len(t)):
        V[idx] = np.sin(t[idx])  # Example input voltage
        y = odeint(motor.motor_model, y, [t[idx-1], t[idx]], args=(V[idx],))[1]
        i[idx], omega[idx] = y
        
        # Integrate omega to get the actual position
        position_actual[idx] = position_actual[idx-1] + omega[idx] * dt
        
        # Compute position error
        position_error = position_desired[idx] - position_actual[idx]

        # Collect input-output pairs for the neural network
        if idx >= 2:  # Ensure there are at least 2 past states for inputs
            inputs.append([
                position_error,  # Current position error
                position_desired[idx] - position_actual[idx-1],  # Previous position error
                position_desired[idx] - position_actual[idx-2],  # Error two steps back
                omega[idx], omega[idx-1], omega[idx-2]  # Velocity history
            ])
            labels.append([omega[idx]])  # Target is the current angular velocity

    inputs = np.array(inputs)
    labels = np.array(labels)
   
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def visualize_motor_plant_data(num_samples, noise_levels):
    # Generate data using the DC motor plant simulation
    motor = DCMotorPlant()
    motor.set_noise(**noise_levels)

    # Time vector
    t = np.linspace(0, 10, num_samples)
    dt = t[1] - t[0]

    # Desired trajectory (sine wave for position)
    position_desired = np.sin(t)

    # Simulate the motor
    i = np.zeros_like(t)
    omega = np.zeros_like(t)
    V = np.zeros_like(t)
    position_actual = np.zeros_like(t)
    y = [0.0, 0.0]  # Initial conditions: [current, angular velocity]

    inputs = []
    labels = []
    position_errors = []

    for idx in range(1, len(t)):
        V[idx] = np.sin(t[idx])  # Example input voltage
        y = odeint(motor.motor_model, y, [t[idx-1], t[idx]], args=(V[idx],))[1]
        i[idx], omega[idx] = y
        
        # Integrate omega to get the actual position
        position_actual[idx] = position_actual[idx-1] + omega[idx] * dt
        
        # Compute position error
        position_error = position_desired[idx] - position_actual[idx]
        position_errors.append(position_error)

        # Collect input-output pairs for the neural network
        if idx >= 2:  # Ensure there are at least 2 past states for inputs
            inputs.append([
                position_error,  # Current position error
                position_desired[idx] - position_actual[idx-1],  # Previous position error
                position_desired[idx] - position_actual[idx-2],  # Error two steps back
                omega[idx], omega[idx-1], omega[idx-2]  # Velocity history
            ])
            labels.append([omega[idx]])  # Target is the current angular velocity

    # Plot the results
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, position_desired, label='Desired Position (rad)')
    plt.plot(t, position_actual, label='Actual Position (rad)', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('Position Tracking')
    plt.grid()
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, omega, label='Angular Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function for training the neural network with data from the motor plant
def main():
    # Initialize the network
    net = ActuatorNet()
    trainer = ActuatorNetTrainer()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)  # Added weight_decay for regularization

    # Set noise levels for motor plant simulation
    noise_levels = {
        'V_noise': 0.001,    # Noise level in volts
        'i_noise': 0.001,    # Noise level in amperes
        'omega_noise': 0.0005 # Noise level in radians per second
    }

    # Generate training data
    train_loader = generate_motor_plant_data(num_samples=10000, noise_levels=noise_levels)
    val_loader = generate_motor_plant_data(num_samples=2000, noise_levels=noise_levels)

    # Train the network
    trainer.train_model(net, criterion, optimizer, train_loader, val_loader, num_epochs=30)

def test_data():
    # Set noise levels for motor plant simulation
    noise_levels = {
        'V_noise': 0.001,    # Noise level in volts
        'i_noise': 0.001,    # Noise level in amperes
        'omega_noise': 0.0005 # Noise level in radians per second
    }

    visualize_motor_plant_data(num_samples=1000, noise_levels=noise_levels)

if __name__ == "__main__":
    test_data()
