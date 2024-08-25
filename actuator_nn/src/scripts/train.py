import numpy as np
import pandas as pd
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import wandb

def main():

    model_path = '../weights/best_actuator_model16.pt'
    data_path = '../data/normal2+normal3+contact1.txt'

    # Create the model
    model = ActuatorNet(dropout_rate=0.08)
    trainer = ActuatorNetTrainer(model, device='cuda')

    # Set Wandb params
    project_name = 'actuator-net-training'
    run_name = 'actuator-net-16'
    entity_name = 'alvister88'

    
    # Train the model and get test data
    trained_model, X_test, y_test = trainer.train_model(
        data_path=data_path, 
        lri=0.001, lrf=0.00008, batch_size=32, num_epochs=400, weight_decay=0.01, 
        save_path=model_path, entity_name=entity_name, project_name=project_name, run_name=run_name 
    )

    # Evaluate the model after training
    eval_data_path = '../data/contact1.txt'  # Update this path as needed

    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    
    position_errors, velocities, torques = evaluator.load_data(eval_data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)
    
    evaluator.evaluate_model(X, y, position_errors, velocities, torques)

if __name__ == "__main__":
    main()
