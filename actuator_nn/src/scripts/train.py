import numpy as np
import pandas as pd
import torch
import yaml
from ActuatorNet import ActuatorNet, HISTORY_SIZE, NUM_LAYERS
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import wandb

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config('../configs/train_config.yaml')

    # Set paths
    model_path = config['model']['path']
    train_data = config['data']['train_data_path']
    validation_data = config['data']['validation_data_path']
    eval_data_path = config['data']['eval_data_path']

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    # Create the trainer
    trainer = ActuatorNetTrainer(
        hidden_size=config['model']['hidden_size'], 
        num_layers=config['model']['num_layers'], 
        dropout_rate=config['model']['dropout_rate'], 
        device=device
    )

    # Set Wandb params
    project_name = config['wandb']['project_name']
    run_name = config['wandb']['run_name']
    entity_name = config['wandb']['entity_name']

    # Train the model
    trained_model = trainer.train_model(
        train_data_path=train_data,
        val_data_path=validation_data,
        lri=config['training']['lri'],
        lrf=config['training']['lrf'],
        batch_size=config['training']['batch_size'],
        patience=config['training']['patience'],
        num_epochs=config['training']['num_epochs'],
        pct_start=config['training']['pct_start'],
        anneal_strategy=config['training']['anneal_strategy'],
        weight_decay=config['training']['weight_decay'],
        save_path=model_path,
        entity_name=entity_name,
        project_name=project_name,
        run_name=run_name
    )

    # Evaluate the model after training
    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    position_errors, velocities, currents, torques = evaluator.load_data(eval_data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, currents, torques)

    # Capture the evaluation metrics
    evaluation_metrics = evaluator.evaluate_model(X, y, position_errors, velocities, currents, torques)

if __name__ == "__main__":
    main()
