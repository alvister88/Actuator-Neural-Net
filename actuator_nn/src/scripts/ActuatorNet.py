import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the Actuator Network
class ActuatorNet(nn.Module):
    def __init__(self):
        super(ActuatorNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # Input layer
        self.hidden1_1 = nn.Linear(32, 32)
        self.hidden1_2 = nn.Linear(32, 32)
        self.hidden1_3 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(32, 32)
        self.hidden2_1 = nn.Linear(32, 32)
        self.hidden2_2 = nn.Linear(32, 32)
        self.hidden2_3 = nn.Linear(32, 32)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(32, 32)
        self.hidden3_1 = nn.Linear(32, 32)
        self.hidden3_2 = nn.Linear(32, 32)
        self.hidden3_3 = nn.Linear(32, 32)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(32, 32)
        self.hidden4_1 = nn.Linear(32, 32)
        self.hidden4_2 = nn.Linear(32, 32)
        self.hidden4_3 = nn.Linear(32, 32)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc5 = nn.Linear(32, 32)
        self.hidden5_1 = nn.Linear(32, 32)
        self.hidden5_2 = nn.Linear(32, 32)
        self.hidden5_3 = nn.Linear(32, 32)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc6 = nn.Linear(32, 1)  # Output layer
        
        self.softsign = nn.Softsign()
        
    def forward(self, x):
        # First fully connected layer with 3 hidden layers and dropout
        x = self.softsign(self.fc1(x))
        x = self.softsign(self.hidden1_1(x))
        x = self.softsign(self.hidden1_2(x))
        x = self.softsign(self.hidden1_3(x))
        x = self.dropout1(x)
        
        # Second fully connected layer with 3 hidden layers and dropout
        x = self.softsign(self.fc2(x))
        x = self.softsign(self.hidden2_1(x))
        x = self.softsign(self.hidden2_2(x))
        x = self.softsign(self.hidden2_3(x))
        x = self.dropout2(x)
        
        # Third fully connected layer with 3 hidden layers and dropout
        x = self.softsign(self.fc3(x))
        x = self.softsign(self.hidden3_1(x))
        x = self.softsign(self.hidden3_2(x))
        x = self.softsign(self.hidden3_3(x))
        x = self.dropout3(x)
        
        # Fourth fully connected layer with 3 hidden layers and dropout
        x = self.softsign(self.fc4(x))
        x = self.softsign(self.hidden4_1(x))
        x = self.softsign(self.hidden4_2(x))
        x = self.softsign(self.hidden4_3(x))
        x = self.dropout4(x)
        
        # Fifth fully connected layer with 3 hidden layers and dropout
        x = self.softsign(self.fc5(x))
        x = self.softsign(self.hidden5_1(x))
        x = self.softsign(self.hidden5_2(x))
        x = self.softsign(self.hidden5_3(x))
        x = self.dropout5(x)
        
        # Output layer
        x = self.fc6(x)
        return x

