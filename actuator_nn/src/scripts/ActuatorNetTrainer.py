import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
import time

class ActuatorNetTrainer:
    @staticmethod
    def prepare_sequence_data(position_errors, velocities, torques, sequence_length=3):
        X, y = [], []
        for i in range(len(torques) - sequence_length + 1):
            X.append(np.column_stack((position_errors[i:i+sequence_length], 
                                      velocities[i:i+sequence_length])).flatten())
            y.append(torques[i+sequence_length-1])
        return np.array(X), np.array(y)

    @staticmethod
    def train_model(net, position_errors, velocities, torques, lri=0.001, lrf=0.0001, batch_size=32, num_epochs=1000, 
                    save_path='actuator_model.pt', project_name='actuator-net-training', run_name='actuator-net-run'):
        # Ensure unique save path
        base_save_path = save_path.rsplit('.', 1)[0]
        extension = save_path.rsplit('.', 1)[1] if '.' in save_path else ''
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base_save_path}_{counter}.{extension}"
            counter += 1

        # Set device to GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the selected device
        net.to(device)

        # Print whether using GPU or CPU
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lri)

        # Initialize W&B
        wandb.init(project=project_name, name=run_name, config={
            "learning_rate_initial": lri,
            "learning_rate_final": lrf,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "save_path": save_path,
        })

        # Learning rate scheduler
        lr_lambda = lambda epoch: lrf / lri + (1 - epoch / num_epochs) * (1 - lrf / lri)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        X, y = ActuatorNetTrainer.prepare_sequence_data(position_errors, velocities, torques)
        
        # Split data into train, validation, and test sets
        train_split = int(0.8 * len(X))
        val_split = int(0.9 * len(X))
        
        X_train = torch.FloatTensor(X[:train_split]).to(device)
        y_train = torch.FloatTensor(y[:train_split]).to(device)
        X_val = torch.FloatTensor(X[train_split:val_split]).to(device)
        y_val = torch.FloatTensor(y[train_split:val_split]).to(device)
        X_test = torch.FloatTensor(X[val_split:]).to(device)
        y_test = torch.FloatTensor(y[val_split:]).to(device)

        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        total_train_time = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            net.train()
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = net(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            epoch_loss /= (len(X_train) // batch_size)
            train_losses.append(epoch_loss)

            # Evaluate on validation set
            net.eval()
            with torch.no_grad():
                val_outputs = net(X_val)
                val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
                val_losses.append(val_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            total_train_time += epoch_duration

            # Log metrics to W&B
            wandb.log({
                "Train Loss": epoch_loss,
                "Val Loss": val_loss,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Epoch": epoch + 1,
                "Epoch Duration (seconds)": epoch_duration,
                "Total Training Time (seconds)": total_train_time
            })

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), save_path)
                wandb.run.summary["Best Val Loss"] = best_val_loss
                wandb.save(os.path.basename(save_path))  # Save model checkpoint to W&B using basename
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f} (Best - Saved to {save_path})')
            elif (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')

            # Step the scheduler
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr()[0])

        # Log final training time
        wandb.run.summary["Total Training Time (seconds)"] = total_train_time

        # Close W&B run
        wandb.finish()

        return net, X_test, y_test

    @staticmethod
    def load_model(net, run_device='cpu', load_path='actuator_model.pt'):
        # Load model weights
        device = run_device
        net.to(device)  # Move model to GPU

        if os.path.exists(load_path):
            net.load_state_dict(torch.load(load_path))
            print(f'Model loaded from {load_path}')
        else:
            print(f'No saved model found at {load_path}')
        return net

    @staticmethod
    def evaluate_model(net, X_test, y_test, position_errors, velocities, torques):
        # Evaluate model
        device = 'cpu'
        net.to(device)
        net.eval()

        with torch.no_grad():
            predictions = net(X_test).cpu().numpy().flatten()  # Move predictions to CPU before converting to numpy

        # Move y_test to CPU and convert to numpy
        y_test_np = y_test.cpu().numpy()

        # Calculate RMS error
        rms_error = np.sqrt(np.mean((predictions - y_test_np) ** 2))
        print(f'Test RMS Error: {rms_error:.3f} N·m')

        # Plot predictions vs actual
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test_np, predictions, alpha=0.5)
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
        plt.xlabel('Actual Torque (N·m)')
        plt.ylabel('Predicted Torque (N·m)')
        plt.title('Actuator Network Predictions vs Actual Torque (Test Set)')
        plt.show()

        # Plot error, velocity, predicted torque, and actual torque
        test_start = int(0.9 * len(position_errors))
        test_errors = position_errors[test_start:]
        test_velocities = velocities[test_start:]

        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(test_errors, label='Error')
        plt.title('Test Data Visualization')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(test_velocities, label='Velocity')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(y_test_np, label='Actual Torque')
        plt.plot(predictions, label='Predicted Torque')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(y_test_np - predictions, label='Prediction Error')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return predictions