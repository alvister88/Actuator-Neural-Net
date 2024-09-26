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
    prediction_files = ['predictions/predicted_torque_actuator_gruv3_model13.pt.txt',
                        'predictions/predicted_torque_actuator_gruv2_model18.pt.txt',
                        'predictions/tuned_MLP.txt']
    model_names = ['PVA-GRU', 'PV-GRU', 'MLP (Tuned)']

    evaluator = ActuatorNetEvaluator(model_path, run_device='cuda')
    
    evaluator.plot_predictions(data_file=data_path, prediction_files=prediction_files, model_names=model_names, plot_vs_time=True, save_html=False)
    evaluator.plot_error_histograms(data_file=data_path, prediction_files=prediction_files, model_names=model_names, save_html=False)

if __name__ == "__main__":
    main()