import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ActuatorNet import ActuatorNet

class ActuatorNetTrainer:
    @staticmethod
    def train_model(net, criterion, optimizer, train_loader, val_loader=None, num_epochs=100, save_path='actuator_net_weights.pt'):
        train_losses = []
        val_losses = []

        # Store predictions and actual values
        all_train_preds = []
        all_train_labels = []
        all_val_preds = []
        all_val_labels = []

        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            train_preds = []
            train_labels = []

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                
                optimizer.zero_grad()
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Store predictions and labels for plotting
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.3f}')
            
            all_train_preds.extend(train_preds)
            all_train_labels.extend(train_labels)
            
            if val_loader:
                net.eval()
                val_loss = 0.0
                val_preds = []
                val_labels = []

                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        # Store predictions and labels for plotting
                        val_preds.extend(outputs.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                epoch_val_loss = val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)
                print(f'Epoch {epoch + 1}, Validation Loss: {epoch_val_loss:.3f}')
                
                all_val_preds.extend(val_preds)
                all_val_labels.extend(val_labels)
        
        print('Finished Training')
        
        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        if val_loader:
            plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
        
        # Plot the trajectories and predicted vs actual torque
        plt.figure(figsize=(10, 5))
        plt.plot(all_train_labels, label='Actual Torque (Training)')
        plt.plot(all_train_preds, label='Predicted Torque (Training)')
        if val_loader:
            plt.plot(all_val_labels, label='Actual Torque (Validation)')
            plt.plot(all_val_preds, label='Predicted Torque (Validation)')
        plt.xlabel('Samples')
        plt.ylabel('Torque')
        plt.legend()
        plt.title('Predicted vs Actual Torque')
        plt.show()

        # Save the model weights
        torch.save(net.state_dict(), save_path)
        print(f"Model weights saved to {save_path}.")
