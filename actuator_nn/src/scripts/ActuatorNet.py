import torch
import torch.nn as nn
import torch.optim as optim

# Define the Actuator Network
class ActuatorNet(nn.Module):
    def __init__(self):
        super(ActuatorNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # Input layer
        self.hidden1_1 = nn.Linear(32, 32)
        self.hidden1_2 = nn.Linear(32, 32)
        self.hidden1_3 = nn.Linear(32, 32)
        
        self.fc2 = nn.Linear(32, 32)
        self.hidden2_1 = nn.Linear(32, 32)
        self.hidden2_2 = nn.Linear(32, 32)
        self.hidden2_3 = nn.Linear(32, 32)
        
        self.fc3 = nn.Linear(32, 32)
        self.hidden3_1 = nn.Linear(32, 32)
        self.hidden3_2 = nn.Linear(32, 32)
        self.hidden3_3 = nn.Linear(32, 32)
        
        self.fc4 = nn.Linear(32, 32)
        self.hidden4_1 = nn.Linear(32, 32)
        self.hidden4_2 = nn.Linear(32, 32)
        self.hidden4_3 = nn.Linear(32, 32)
        
        self.fc5 = nn.Linear(32, 32)
        self.hidden5_1 = nn.Linear(32, 32)
        self.hidden5_2 = nn.Linear(32, 32)
        self.hidden5_3 = nn.Linear(32, 32)
        
        self.fc6 = nn.Linear(32, 1)  # Output layer
        
        self.softsign = nn.Softsign()
        
    def forward(self, x):
        # First fully connected layer with 3 hidden layers
        x = self.softsign(self.fc1(x))
        x = self.softsign(self.hidden1_1(x))
        x = self.softsign(self.hidden1_2(x))
        x = self.softsign(self.hidden1_3(x))
        
        # Second fully connected layer with 3 hidden layers
        x = self.softsign(self.fc2(x))
        x = self.softsign(self.hidden2_1(x))
        x = self.softsign(self.hidden2_2(x))
        x = self.softsign(self.hidden2_3(x))
        
        # Third fully connected layer with 3 hidden layers
        x = self.softsign(self.fc3(x))
        x = self.softsign(self.hidden3_1(x))
        x = self.softsign(self.hidden3_2(x))
        x = self.softsign(self.hidden3_3(x))
        
        # Fourth fully connected layer with 3 hidden layers
        x = self.softsign(self.fc4(x))
        x = self.softsign(self.hidden4_1(x))
        x = self.softsign(self.hidden4_2(x))
        x = self.softsign(self.hidden4_3(x))
        
        # Fifth fully connected layer with 3 hidden layers
        x = self.softsign(self.fc5(x))
        x = self.softsign(self.hidden5_1(x))
        x = self.softsign(self.hidden5_2(x))
        x = self.softsign(self.hidden5_3(x))
        
        # Output layer
        x = self.fc6(x)
        return x
    
    # Example training loop
    def train_model(net, criterion, optimizer, train_loader, num_epochs=100):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                
                optimizer.zero_grad()
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')