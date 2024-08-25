import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import wandb
import time
import pandas as pd
from ActuatorNet import ActuatorNet

class ActuatorNetTrainer:
    def __init__(self, net, device=None):
        self.net = net
        self.device = self.setup_device(device)
        self.net.to(self.device)

    def load_data(self, file_path):
        data = pd.read_csv(file_path, delimiter=',')
        position_errors = data['Error'].values
        velocities = data['Velocity'].values
        torques = data['Torque'].values
        return position_errors, velocities, torques
    
    def prepare_sequence_data(self, position_errors, velocities, torques, sequence_length=3):
        X, y = [], []
        for i in range(len(torques) - sequence_length + 1):
            X.append(np.column_stack((position_errors[i:i+sequence_length], 
                                      velocities[i:i+sequence_length])).flatten())
            y.append(torques[i+sequence_length-1])
        return np.array(X), np.array(y)

    def setup_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        return device

    def prepare_data(self, position_errors, velocities, torques):
        X, y = self.prepare_sequence_data(position_errors, velocities, torques)
        train_split, val_split = int(0.8 * len(X)), int(0.9 * len(X))
        X_train, y_train = torch.FloatTensor(X[:train_split]).to(self.device), torch.FloatTensor(y[:train_split]).to(self.device)
        X_val, y_val = torch.FloatTensor(X[train_split:val_split]).to(self.device), torch.FloatTensor(y[train_split:val_split]).to(self.device)
        X_test, y_test = torch.FloatTensor(X[val_split:]).to(self.device), torch.FloatTensor(y[val_split:]).to(self.device)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def setup_training(self, lri, lrf, weight_decay, num_epochs):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=lri, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda epoch: lrf / lri + (1 - epoch / num_epochs) * (1 - lrf / lri))
        return criterion, optimizer, scheduler

    def train_epoch(self, X_train, y_train, optimizer, criterion, batch_size):
        self.net.train()
        train_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X, batch_y = X_train[i:i+batch_size], y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = self.net(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / (len(X_train) // batch_size)

    def validate(self, X_val, y_val, criterion):
        self.net.eval()
        with torch.no_grad():
            val_outputs = self.net(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
        return val_loss

    def log_metrics(self, epoch, num_epochs, train_loss, val_loss, lr, epoch_duration, total_train_time, is_best, save_path):
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Learning Rate": lr,
            "Epoch": epoch + 1,
            "Epoch Duration (seconds)": epoch_duration,
            "Total Training Time (seconds)": total_train_time
        })
        if is_best or (epoch + 1) % 50 == 0:
            message = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}'
            if is_best:
                message += f' (Best - Saved to {save_path})'
            print(message)

    def save_best_model(self, val_loss, save_path):
        if val_loss < wandb.run.summary.get("Best Val Loss", float('inf')):
            torch.save(self.net.state_dict(), save_path)
            wandb.run.summary["Best Val Loss"] = val_loss
            wandb.save(os.path.basename(save_path))
            return True
        return False

    def ensure_unique_save_path(self, save_path):
        base_save_path = save_path.rsplit('.', 1)[0]
        extension = save_path.rsplit('.', 1)[1] if '.' in save_path else ''
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base_save_path}_{counter}.{extension}"
            counter += 1
        return save_path

    def train_model(self, data_path='../data/normal1', lri=0.001, lrf=0.0001, batch_size=32, 
                    weight_decay=0.01, num_epochs=1000, save_path='actuator_model.pt', 
                    entity_name='your_account', project_name='actuator-net-training', run_name='actuator-net-run'):
        
        save_path = self.ensure_unique_save_path(save_path)
        position_errors, velocities, torques = self.load_data(data_path)

        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(position_errors, velocities, torques)
        criterion, optimizer, scheduler = self.setup_training(lri, lrf, weight_decay, num_epochs)

        wandb.init(project=project_name, entity=entity_name, name=run_name, config={
            "learning_rate_initial": lri,
            "learning_rate_final": lrf,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "save_path": save_path,
            "weight_decay": weight_decay
        })

        total_train_time = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(X_train, y_train, optimizer, criterion, batch_size)
            val_loss = self.validate(X_val, y_val, criterion)

            epoch_duration = time.time() - epoch_start_time
            total_train_time += epoch_duration

            is_best = self.save_best_model(val_loss, save_path)
            if is_best:
                best_val_loss = val_loss

            self.log_metrics(epoch, num_epochs, train_loss, val_loss, scheduler.get_last_lr()[0], 
                             epoch_duration, total_train_time, is_best, save_path)

            scheduler.step()

        wandb.finish()
        return self.net, X_test, y_test