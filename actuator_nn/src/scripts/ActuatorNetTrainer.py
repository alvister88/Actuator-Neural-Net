import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import wandb
import time
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE

class ActuatorNetTrainer:
    def __init__(self, hidden_size=64, num_layers=2, dropout_rate=0.2, device=None):
        self.device = self.setup_device(device)
        self.net = ActuatorNet(hidden_size, num_layers, dropout_rate).to(self.device)

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

    def load_data(self, file_path):
        data = pd.read_csv(file_path, delimiter=',')
        position_errors = data['Error'].values
        velocities = data['Velocity'].values
        torques = data['Torque'].values
        return position_errors, velocities, torques

    def prepare_sequence_data(self, position_errors, velocities, torques):
        X, y = [], []
        for i in range(len(torques) - HISTORY_SIZE + 1):
            X.append(np.column_stack((position_errors[i:i+HISTORY_SIZE], 
                                      velocities[i:i+HISTORY_SIZE])))
            y.append(torques[i+HISTORY_SIZE-1])
        return np.array(X), np.array(y)

    def prepare_data(self, train_data_path, val_data_path):
        train_position_errors, train_velocities, train_torques = self.load_data(train_data_path)
        val_position_errors, val_velocities, val_torques = self.load_data(val_data_path)

        X_train, y_train = self.prepare_sequence_data(train_position_errors, train_velocities, train_torques)
        X_val, y_val = self.prepare_sequence_data(val_position_errors, val_velocities, val_torques)

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        return X_train, y_train, X_val, y_val

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

    def train_model(self, train_data_path, val_data_path, 
                    lri=0.001, lrf=0.0001, batch_size=32, patience=50,
                    weight_decay=0.01, num_epochs=1000, save_path='actuator_model_gru.pt', 
                    entity_name='your_account', project_name='actuator-net-training', run_name='actuator-net-gru-run'):  

        save_path = self.ensure_unique_save_path(save_path)
        X_train, y_train, X_val, y_val = self.prepare_data(train_data_path, val_data_path)
        criterion, optimizer, scheduler = self.setup_training(lri, lrf, weight_decay, num_epochs)

        wandb.init(project=project_name, entity=entity_name, name=run_name, config={
            "learning_rate_initial": lri,
            "learning_rate_final": lrf,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "save_path": save_path,
            "weight_decay": weight_decay,
            "patience": patience,
            "hidden_size": self.net.gru.hidden_size,
            "num_layers": self.net.gru.num_layers,
            "dropout_rate": self.net.dropout.p
        })

        total_train_time = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(X_train, y_train, optimizer, criterion, batch_size)
            val_loss = self.validate(X_val, y_val, criterion)

            epoch_duration = time.time() - epoch_start_time
            total_train_time += epoch_duration

            is_best = self.save_best_model(val_loss, save_path)
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.log_metrics(epoch, num_epochs, train_loss, val_loss, scheduler.get_last_lr()[0], 
                             epoch_duration, total_train_time, is_best, save_path)

            scheduler.step()

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        wandb.finish()
        return self.net