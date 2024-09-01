import torch
import torch.nn as nn

# GLOBAL VARIABLES
HISTORY_SIZE = 28
INPUT_SIZE = 2  # Position error and velocity
NUM_LAYERS = 4

class ActuatorNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HISTORY_SIZE, num_layers=NUM_LAYERS, dropout_rate=0.2):
        super(ActuatorNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Reshape input: (batch_size, INPUT_SIZE * HISTORY_SIZE) -> (batch_size, HISTORY_SIZE, INPUT_SIZE)
        x = x.view(-1, self.hidden_size, self.input_size)
        
        # Pass through GRU
        out, _ = self.gru(x)
        
        # We only need the last output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Final fully connected layer
        out = self.fc(out)
        
        return out