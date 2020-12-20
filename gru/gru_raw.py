import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size.
        
        self.rt_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.zt_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input, hidden):
        
        batch = input.size(0)
        
        input = input.view(batch, self.input_size)
        hidden = hidden.view(batch, self.input_size)
        
        combine = input + hidden
        
        rt = self.sigmoid(self.rt_layer(combine))
        zt = self.sigmoid(self.zt_layer(combine))
        
        nt = self.tanh(self.input_layer(input) + rt * self.hidden_layer(hidden))
        
        oneone = torch.ones(batch, self.hidden_size)
        
        hidden_state = (oneone - zt) * nt + zt * hidden
        
        return hidden_state