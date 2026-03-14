import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl

def extract_global_features(df, column_name):
    """Provides 6 key signals about a column to the brain."""
    col = df[column_name]
    features = []
    # 1. Is it numeric?
    features.append(1.0 if col.dtype in [pl.Float64, pl.Int64] else 0.0)
    # 2. How many nulls?
    features.append(col.null_count() / len(df))
    # 3. Currency/Symbol detection
    sample = str(col.slice(0, 10).to_list())
    features.append(1.0 if any(s in sample for s in ['$', '€', '£']) else 0.0)
    features.append(1.0 if '%' in sample else 0.0)
    features.append(1.0 if any(s in sample for s in ['-', '/']) else 0.0)
    # 4. Skewness (Outliers)
    if col.dtype in [pl.Float64, pl.Int64]:
        skew = col.skew() if col.skew() is not None else 0.0
        features.append(min(abs(skew) / 5.0, 1.0))
    else:
        features.append(0.0)
    return features

class Mind1Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mind1Net, self).__init__()
        # 3-layer architecture for data pattern recognition
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Mind1Agent: # Corrected spelling from 'Agnet'
    def __init__(self, input_size, output_size):
        # Corrected: Initialize the Network, not the Agent class itself
        self.model = Mind1Net(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        """Brain chooses the action with the highest expected reward."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()