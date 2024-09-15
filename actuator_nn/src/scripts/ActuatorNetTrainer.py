import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR 
import numpy as np
import pandas as pd
import os
import wandb
import time
from ActuatorNet import ActuatorNet, HISTORY_SIZE, INPUT_SIZE, MAX_ERROR, MAX_VELOCITY, MAX_TORQUE
import signal

class ActuatorNetTrainer:
    def __init__(self, input_size=INPUT_SIZE, device=None):
        self.device = self.setup_device(device)
        self.net = ActuatorNet().to(self.device)
        self.stop_training = False

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
    
    def normalize_data(self, data, min_val, max_val):
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def prepare_data(self, train_data_path, val_data_path):
        train_position_errors, train_velocities, train_torques = self.load_data(train_data_path)
        val_position_errors, val_velocities, val_torques = self.load_data(val_data_path)

        train_position_errors = self.normalize_data(train_position_errors, -MAX_ERROR, MAX_ERROR)
        train_velocities = self.normalize_data(train_velocities, -MAX_VELOCITY, MAX_VELOCITY)
        train_torques = self.normalize_data(train_torques, -MAX_TORQUE, MAX_TORQUE)

        val_position_errors = self.normalize_data(val_position_errors, -MAX_ERROR, MAX_ERROR)
        val_velocities = self.normalize_data(val_velocities, -MAX_VELOCITY, MAX_VELOCITY)
        val_torques = self.normalize_data(val_torques, -MAX_TORQUE, MAX_TORQUE)

        X_train, y_train = self.prepare_sequence_data(train_position_errors, train_velocities, train_torques)
        X_val, y_val = self.prepare_sequence_data(val_position_errors, val_velocities, val_torques)

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        return X_train, y_train, X_val, y_val

    def denormalize_torque(self, normalized_torque):
        return (normalized_torque + 1) * (2 * MAX_TORQUE) / 2 - MAX_TORQUE

    def setup_training(self, lri, lrf, weight_decay, num_epochs, steps_per_epoch, pct_start, anneal_strategy):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=lri, weight_decay=weight_decay)
        total_steps = num_epochs * steps_per_epoch
        scheduler = OneCycleLR(optimizer, 
                            max_lr=lri,
                            total_steps=total_steps,
                            pct_start=pct_start,
                            anneal_strategy=anneal_strategy,
                            final_div_factor=lri/lrf,
                            div_factor=25)
        return criterion, optimizer, scheduler

    def calculate_rms_error(self, predictions, targets):
        denorm_predictions = self.denormalize_torque(predictions)
        denorm_targets = self.denormalize_torque(targets)
        return torch.sqrt(torch.mean((denorm_predictions - denorm_targets) ** 2)).item()

    def train_epoch(self, X_train, y_train, optimizer, criterion, scheduler, batch_size):
        self.net.train()
        train_loss = 0
        train_rms_error = 0
        num_batches = len(X_train) // batch_size
        for i in range(num_batches):
            batch_X = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten to shape (batch_size, INPUT_SIZE)

            optimizer.zero_grad()
            outputs = self.net(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_rms_error += self.calculate_rms_error(outputs, batch_y.unsqueeze(1))
        return train_loss / num_batches, train_rms_error / num_batches

    def validate(self, X_val, y_val, criterion):
        self.net.eval()
        with torch.no_grad():
            X_val = X_val.view(X_val.size(0), -1)  # Flatten to shape (batch_size, INPUT_SIZE)
            
            val_outputs = self.net(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
            val_rms_error = self.calculate_rms_error(val_outputs, y_val.unsqueeze(1))
        return val_loss, val_rms_error


    def log_metrics(self, epoch, num_epochs, train_loss, train_rms, val_loss, val_rms, lr, epoch_duration, total_train_time, is_best, save_path):
        wandb.log({
            "Train Loss": train_loss,
            "Train RMS Error": train_rms,
            "Val Loss": val_loss,
            "Val RMS Error": val_rms,
            "Learning Rate": lr,
            "Epoch": epoch + 1,
            "Epoch Duration (seconds)": epoch_duration,
            "Total Training Time (seconds)": total_train_time
        })
        if is_best or (epoch + 1) % 50 == 0:
            message = (f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Train Loss: {train_loss:.7f}, Train RMS: {train_rms:.7f}, '
                       f'Val Loss: {val_loss:.7f}, Val RMS: {val_rms:.7f}, '
                       f'LR: {lr:.7f}')
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

    def handle_interrupt(self, signum, frame):
        print("\nInterrupt received. Stopping training after this epoch...")
        self.stop_training = True

    def train_model(self, train_data_path, val_data_path, 
                    lri=0.01, lrf=0.001, batch_size=64, patience=100,
                    weight_decay=0.01, num_epochs=1000, pct_start=0.2, anneal_strategy='cos', save_path='actuator_model.pt', 
                    entity_name='your_account', project_name='actuator-net-training', run_name='actuator-net-run'):  
        """
        Trains the actuator neural network model using the provided training and validation data.

        Args:
            train_data_path (str): Path to the training data file.
            val_data_path (str): Path to the validation data file.
            lri (float, optional): Initial learning rate. Defaults to 0.01.
            lrf (float, optional): Final learning rate. Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            patience (int, optional): Number of epochs without improvement before early stopping. Defaults to 100.
            weight_decay (float, optional): Weight decay for regularization. Defaults to 0.01.
            num_epochs (int, optional): Total number of training epochs. Defaults to 1000.
            pct_start (float, optional): Percentage of the cycle spent increasing the learning rate. Defaults to 0.2.
            anneal_strategy (str, optional): Annealing strategy for the learning rate. Defaults to 'cos'.
            save_path (str, optional): Path to save the trained model. Defaults to 'actuator_model.pt'.
            entity_name (str, optional): Entity name for Weights & Biases. Defaults to 'your_account'.
            project_name (str, optional): Project name for Weights & Biases. Defaults to 'actuator-net-training'.
            run_name (str, optional): Run name for Weights & Biases. Defaults to 'actuator-net-run'.

        Returns:
            torch.nn.Module: The trained actuator neural network model.
        """
        save_path = self.ensure_unique_save_path(save_path)
        X_train, y_train, X_val, y_val = self.prepare_data(train_data_path, val_data_path)
        
        steps_per_epoch = len(X_train) // batch_size
        criterion, optimizer, scheduler = self.setup_training(lri, lrf, weight_decay, num_epochs, steps_per_epoch, pct_start, anneal_strategy)

        wandb.init(project=project_name, entity=entity_name, name=run_name, config={
            "learning_rate_max": lri,
            "learning_rate_final": lrf,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "pct_start": pct_start,
            "anneal_strategy": anneal_strategy,
            "save_path": save_path,
            "weight_decay": weight_decay,
            "patience": patience,
            "hidden_size": 32,
            "num_layers": 3,
            "activation": "softsign"
        })

        total_train_time = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Set up the interrupt handler
        signal.signal(signal.SIGINT, self.handle_interrupt)

        print("Training started. Press Ctrl+C to stop training manually.")

        for epoch in range(num_epochs):
            if self.stop_training:
                print("Manual early stopping triggered.")
                break

            epoch_start_time = time.time()

            train_loss, train_rms = self.train_epoch(X_train, y_train, optimizer, criterion, scheduler, batch_size)
            val_loss, val_rms = self.validate(X_val, y_val, criterion)

            epoch_duration = time.time() - epoch_start_time
            total_train_time += epoch_duration

            is_best = self.save_best_model(val_loss, save_path)
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.log_metrics(epoch, num_epochs, train_loss, train_rms, val_loss, val_rms, 
                             optimizer.param_groups[0]['lr'], epoch_duration, total_train_time, is_best, save_path)

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        wandb.finish()
        return self.net

# Example usage
if __name__ == "__main__":
    trainer = ActuatorNetTrainer()
    trainer.train_model('path/to/train_data.csv', 'path/to/val_data.csv')