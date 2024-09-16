import numpy as np
import pandas as pd
import torch
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import wandb

def main():
    # Define file paths for data and model weights
    model_path = '../weights/mlp_actuator_model9.pt'
    train_data_path = '../data/train_data_6.txt'
    validation_data_path = '../data/validation_data_6.txt'
    eval_data_path = '../data/validation_data_6.txt'  # For evaluation after training

    # Initialize model and define training device
    model = ActuatorNet(dropout_rate=0.1)  # Adjust dropout as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer object
    trainer = ActuatorNetTrainer(model, device=device)

    # Wandb project configuration
    project_name = 'actuator-net-training'
    run_name = 'mlp_actuator-net-9'
    entity_name = 'alvister88'

    # Wandb initialization
    wandb.init(project=project_name, entity=entity_name, name=run_name)

    # Train the model using specified parameters
    trained_model = trainer.train_model(
        train_data_path=train_data_path,
        val_data_path=validation_data_path,
        lri=0.0001,            # Initial learning rate
        lrf=0.000008,           # Final learning rate
        batch_size=64,          # Batch size for training
        patience=300,           # Patience for early stopping
        num_epochs=4000,         # Number of epochs to train
        weight_decay=0.01,      # Weight decay for regularization
        save_path=model_path,   # Path to save the trained model
        entity_name=entity_name,
        project_name=project_name,
        run_name=run_name
    )

    # Perform evaluation after training
    evaluator = ActuatorNetEvaluator(model_path=model_path, run_device='cpu')

    # Load evaluation data
    position_errors, velocities, accelerations, torques = evaluator.load_data(eval_data_path)

    # Prepare input sequences for evaluation
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, accelerations, torques)

    # Evaluate the trained model
    evaluation_metrics = evaluator.evaluate_model(
        X, y, 
        position_errors, velocities, accelerations, torques, 
        vs_time=False,     # Set to True if you want to plot vs time
        save_html=False,   # Save plots as HTML
        save_pdf=False,    # Save plots as PDFs
        pdf_subplots=None, # Specify subplots to save as PDF
        save_predictions=False, # Save the predictions
        prediction_output_file=None # Path to save predictions, if needed
    )
    # Finish the Wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
