import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ActuatorNet import ActuatorNet
from ActuatorNetTrainer import ActuatorNetTrainer
from ActuatorNetEvaluator import ActuatorNetEvaluator
import time


def main():
    # Load the dataset
    data_path = '../data/gains3/test3.txt'  # Update this path as needed
    model_path = '../weights/actuator_gruv3_model13.pt'  # Update this path as needed

    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    
    position_errors, velocities, accelerations, torques = evaluator.load_data(data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, accelerations, torques)
    save_plots = ['torque', 'model_error']
    
    evaluator.evaluate_model(X, y, position_errors, velocities, accelerations, torques, 
                             vs_time=True, save_html=False, save_pdf=False, pdf_subplots=save_plots, 
                             save_predictions=False)

if __name__ == "__main__":
    main()