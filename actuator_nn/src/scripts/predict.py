import torch
import numpy as np
import matplotlib.pyplot as plt
from ActuatorNet import ActuatorNet
from actuator_nn.src.utils.DCMotorPlant import DCMotorPlant
from scipy.integrate import odeint

# Load the model weights
def load_model_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()  # Set to evaluation mode
    print("Model weights loaded.")

# Generate test data using the simulated motor plant
def generate_motor_plant_test_data(num_samples, noise_levels=None):
    motor = DCMotorPlant()
    if noise_levels:
        motor.set_noise(**noise_levels)

    # Time vector
    t = np.linspace(0, 2 * np.pi, num_samples)
    dt = t[1] - t[0]
    
    # Desired angular velocity (sine wave)
    omega_desired = np.sin(t)
    
    # Simulate the motor with a PID controller
    i = np.zeros_like(t)
    omega = np.zeros_like(t)
    V = np.zeros_like(t)
    y = [0.0, 0.0]  # Initial conditions: [current, angular velocity]

    for idx in range(1, len(t)):
        # No PID in this testing, we simulate the motor response directly
        V[idx] = omega_desired[idx]  # Use the desired angular velocity as a direct input for simplicity
        y = odeint(motor.motor_model, y, [t[idx-1], t[idx]], args=(V[idx],))[1]
        i[idx], omega[idx] = y

    # Create input features: current state and past two states
    inputs = []
    actuals = []
    for i in range(2, num_samples):
        inputs.append([
            omega[i], omega[i-1], omega[i-2],
            V[i], V[i-1], V[i-2]
        ])
        actuals.append(omega_desired[i])  # Actual desired torque

    inputs = np.array(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    actuals = np.array(actuals)
    return inputs, actuals

def main():
    # Initialize the network
    net = ActuatorNet()

    # Load the saved weights
    load_model_weights(net, '../weights/actuator_net_weights.pt')

    # Generate test data using the motor plant
    test_inputs, actual_torques = generate_motor_plant_test_data(100, noise_levels={'V_noise': 0.01, 'i_noise': 0.01, 'omega_noise': 0.05})

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
