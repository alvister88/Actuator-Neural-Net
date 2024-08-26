import numpy as np
import pandas as pd
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import wandb

def main():
    model_path = '../weights/best_actuator_model26.pt'
    train_data = '../data/train_data1.txt'
    validation_data = '../data/validation_data1.txt'

    # Create the model
    model = ActuatorNet(dropout_rate=0.1)

    # Create the trainer
    trainer = ActuatorNetTrainer(model, device='cuda')

    # Set Wandb params
    project_name = 'actuator-net-training'
    run_name = 'actuator-net-26'
    entity_name = 'alvister88'

    # Train the model and get test data
    trained_model = trainer.train_model(
        train_data_path=train_data, val_data_path=validation_data,
        lri=0.00005, lrf=0.000005, batch_size=32, patience=100, num_epochs=800, weight_decay=0.01,
        save_path=model_path, entity_name=entity_name, project_name=project_name, run_name=run_name
    )

    # Evaluate the model after training
    eval_data_path = '../data/normal1.txt'  # Update this path as needed
    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    position_errors, velocities, torques = evaluator.load_data(eval_data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)

    # Capture the evaluation metrics
    evaluation_metrics = evaluator.evaluate_model(X, y, position_errors, velocities, torques)

if __name__ == "__main__":
    main()