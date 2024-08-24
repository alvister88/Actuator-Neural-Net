import numpy as np
import pandas as pd
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    position_errors = data['Error'].values
    velocities = data['Velocity'].values
    torques = data['Torque'].values
    return position_errors, velocities, torques

def main():
    # Load data
    position_errors, velocities, torques = load_data('../data/normal1.txt')

    # Create and train the model
    model = ActuatorNet(dropout_rate=0.2)
    trained_model, X_val, y_val = ActuatorNetTrainer.train_model(model, position_errors, velocities, torques, learning_rate=0.001, batch_size=32, num_epochs=2000)

    # Evaluate the model
    ActuatorNetTrainer.evaluate_model(trained_model, X_val, y_val, position_errors, velocities, torques)

if __name__ == "__main__":
    main()