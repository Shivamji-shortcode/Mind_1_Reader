import torch
import torch.nn as nn
import torch.optim as optim

class Mind1Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mind1Net, self).__init__()
        # A simple but powerful 3-layer neural network 
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size) # 3 output for out three actions
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Mind1Agnet:
    def __init__(self, input_size, output_size):
        self.model = Mind1Agnet(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        # For nwo we are goinf to let the agent to pick the highest predicted   reward
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()