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
    data_path = '../data/test3.txt'
    model_path = '../weights/mlp_actuator_model5.pt'

    evaluator = ActuatorNetEvaluator(model_path, run_device='cpu')
    
    position_errors, velocities, torques = evaluator.load_data(data_path)
    X, y = evaluator.prepare_sequence_data(position_errors, velocities, torques)
    
    predictions, y_denorm, rms_error, percentage_accuracy, total_inference_time, average_inference_time = evaluator.evaluate_model(X, y, save_predictions=True, prediction_output_file='tuned_MLP.txt')
    
    evaluator.plot_predictions_vs_actual(y_denorm, predictions, save_html=False)
    evaluator.plot_data_visualization(y_denorm, predictions, position_errors, velocities, rms_error, 
                                      percentage_accuracy, total_inference_time, average_inference_time, 
                                      plot_vs_time=False, save_html=False)

if __name__ == "__main__":
    main()