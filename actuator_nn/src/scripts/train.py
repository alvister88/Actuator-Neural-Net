import numpy as np
import pandas as pd
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
import wandb

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    position_errors = data['Error'].values
    velocities = data['Velocity'].values
    torques = data['Torque'].values
    return position_errors, velocities, torques

def main():
    # Load data
    position_errors, velocities, torques = load_data('../data/normal2+normal3+contact1.txt')

    path = '../weights/best_actuator_model11.pt'

    # Create the model
    model = ActuatorNet(dropout_rate=0.1)

    # Set Wandb params
    project_name = 'actuator-net-training'
    run_name = 'actuator-net-11'

    # Train the model and get test data
    trained_model, X_test, y_test = ActuatorNetTrainer.train_model(
        model, position_errors, velocities, torques, 
        lri=0.004, lrf=0.0002, batch_size=32, num_epochs=1000, weight_decay=0.01, 
        save_path=path, project_name=project_name, run_name=run_name 
    )

    # Load the best model
    best_model = ActuatorNetTrainer.load_model(model, load_path=path)

    # Evaluate the best model on the test set
    ActuatorNetTrainer.evaluate_model(best_model, X_test, y_test, position_errors, velocities, torques)

if __name__ == "__main__":
    main()
