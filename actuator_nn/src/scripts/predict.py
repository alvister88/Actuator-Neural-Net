import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ActuatorNet import ActuatorNet  # Make sure this import works with your file structure
from ActuatorNetTrainer import ActuatorNetTrainer

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    position_errors = data['Error'].values
    velocities = data['Velocity'].values
    torques = data['Torque'].values
    return position_errors, velocities, torques

def prepare_sequence_data(position_errors, velocities, torques, sequence_length=3):
    X, y = [], []
    for i in range(len(torques) - sequence_length + 1):
        X.append(np.column_stack((position_errors[i:i+sequence_length], 
                                  velocities[i:i+sequence_length])).flatten())
        y.append(torques[i+sequence_length-1])
    return np.array(X), np.array(y)

def load_model(model_path, dropout_rate=0.2):
    model = ActuatorNet(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, X, y, position_errors, velocities, torques):
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X)).numpy().flatten()

    # Calculate RMS error
    rms_error = np.sqrt(np.mean((predictions - y) ** 2))
    print(f'Test RMS Error: {rms_error:.3f} N·m')

    # Calculate the range of torque values
    torque_range = np.max(y) - np.min(y)
    
    # Calculate percentage accuracy
    percentage_accuracy = (1 - (rms_error / torque_range)) * 100
    print(f'Percentage Accuracy: {percentage_accuracy:.2f}%')

    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Torque (N·m)')
    plt.ylabel('Predicted Torque (N·m)')
    plt.title('Actuator Network Predictions vs Actual Torque')
    plt.show()

    # Plot error, velocity, predicted torque, and actual torque
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(position_errors[-len(y):], label='Error')
    plt.title('Data Visualization')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(velocities[-len(y):], label='Velocity')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(y, label='Actual Torque')
    plt.plot(predictions, label='Predicted Torque')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(y - predictions, label='Prediction Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return predictions

def main():
    # Load the dataset
    data_path = '../data/contact1.txt'  # Update this path as needed
    position_errors, velocities, torques = load_data(data_path)

    # Prepare the data
    X, y = prepare_sequence_data(position_errors, velocities, torques)

    # Load the trained model
    model_path = '../weights/best_actuator_model6.pt'  # Update this path as needed
    model = load_model(model_path)
    
    # Evaluate the model
    evaluate_model(model, X, y, position_errors, velocities, torques)

if __name__ == "__main__":
    main()
