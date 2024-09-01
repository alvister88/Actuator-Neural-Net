import numpy as np
import pandas as pd
import torch
from ActuatorNet import ActuatorNet, HISTORY_SIZE, NUM_LAYERS
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import wandb

def main():
    # Set paths
    model_path = '../weights/best_actuator_gru_model31.pt'
    train_data = '../data/gains2/train_data_3.txt'
    validation_data = '../data/gains2/validation_data_3.txt'
    eval_data_path = '../data/gains2/test1.txt'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Create the model
    # model = ActuatorNet(hidden_size=HISTORY_SIZE, num_layers=NUM_LAYERS, dropout_rate=0.1)

    # Create the trainer
    trainer = ActuatorNetTrainer(hidden_size=HISTORY_SIZE, num_layers=NUM_LAYERS, dropout_rate=0.002, device=device)

    # Set Wandb params
    project_name = 'actuator-net-training'
    run_name = 'actuator-net-gru-31'
    entity_name = 'alvister88'

    # Train the model
    trained_model = trainer.train_model(
        train_data_path=train_data,
        val_data_path=validation_data,
        lri=0.0001,
        lrf=0.00002,
        batch_size=32,
        patience=100,
        num_epochs=1500,
        weight_decay=0.005,
        save_path=model_path,
        entity_name=entity_name,
        project_name=project_name,
        run_name=run_name
    )

    # Evaluate the model after training
    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    position_errors, velocities, torques = evaluator.load_data(eval_data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)

    # Capture the evaluation metrics
    evaluation_metrics = evaluator.evaluate_model(X, y, position_errors, velocities, torques)

if __name__ == "__main__":
    main()