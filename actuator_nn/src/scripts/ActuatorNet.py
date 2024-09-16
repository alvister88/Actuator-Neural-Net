import torch
import torch.nn as nn

# GLOBAL VARIABLES
HISTORY_SIZE = 3  # Based on the document: current state and two past states
INPUT_SIZE = 2 * HISTORY_SIZE  # Position errors and velocities

MAX_TORQUE = 150
MAX_VELOCITY = 250
MAX_ERROR = 6.28

class ActuatorNet(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ActuatorNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.softsign = nn.Softsign()
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer with configurable rate

    def forward(self, x):
        x = self.softsign(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.softsign(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.softsign(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.fc4(x)  # No dropout after the last layer
        return x
