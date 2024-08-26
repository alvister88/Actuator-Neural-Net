import torch
import torch.nn as nn

# GLOBAL VARIABLES
HISTORY_SIZE = 5
INPUT_SIZE = 2 * HISTORY_SIZE

class ActuatorNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ActuatorNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x