import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.ir_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hr_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.iz_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hz_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.in_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hn_layer = nn.Linear(self.hidden_size, self.hidden_size)
              
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input, hidden):
        
        # batch = input.size(0)
        # oneone = torch.ones(batch, self.hidden_size)
        
        r = self.sigmoid(self.ir_layer(input) + self.hr_layer(hidden))
        z = self.sigmoid(self.iz_layer(input) + self.iz_layer(hidden))
        n = self.tanh(self.in_layer(input) + r * self.hn_layer(hidden))
        # hidden_state = (oneone - z) * n + z * hidden
        # hidden_state = (n - n * z) + z * hidden 
        hidden_state = n - z * (n - hidden)
        
        return hidden_state