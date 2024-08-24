import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    def train_model(net, position_errors, velocities, torques, learning_rate=0.001, batch_size=32, num_epochs=1000, validation_split=0.1):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        X, y = ActuatorNetTrainer.prepare_sequence_data(position_errors, velocities, torques)
        
        # Split data into train and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = torch.FloatTensor(X[:split_idx]), torch.FloatTensor(X[split_idx:])
        y_train, y_val = torch.FloatTensor(y[:split_idx]), torch.FloatTensor(y[split_idx:])

        train_losses = []

        for epoch in range(num_epochs):
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

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss over Epochs')
        plt.show()

        # Evaluate on validation set after training
        net.eval()
        with torch.no_grad():
            val_outputs = net(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
        print(f'Validation Loss: {val_loss:.4f}')

        return net, X_val, y_val

    @staticmethod
    def evaluate_model(net, X_val, y_val, position_errors, velocities, torques):
        net.eval()
        with torch.no_grad():
            predictions = net(X_val).numpy().flatten()

        # Calculate RMS error
        rms_error = np.sqrt(np.mean((predictions - y_val.numpy()) ** 2))
        print(f'RMS Error: {rms_error:.3f} N·m')

        # Plot predictions vs actual
        plt.figure(figsize=(10, 5))
        plt.scatter(y_val, predictions, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('Actual Torque (N·m)')
        plt.ylabel('Predicted Torque (N·m)')
        plt.title('Actuator Network Predictions vs Actual Torque')
        plt.show()

        # Plot error, velocity, predicted torque, and actual torque
        val_start = len(position_errors) - len(y_val)
        val_errors = position_errors[val_start:]
        val_velocities = velocities[val_start:]

        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(val_errors, label='Error')
        plt.title('Validation Data Visualization')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(val_velocities, label='Velocity')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(y_val, label='Actual Torque')
        plt.plot(predictions, label='Predicted Torque')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(y_val - predictions, label='Prediction Error')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return predictions