import torch
import torch.nn as nn 

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.forget_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.candidate_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.input_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input, hidden, cell):
        
        combine = input + hidden
        
        ft = self.sigmoid(self.forget_layer(combine))
        ct_ = self.tanh(self.candidate_layer(combine))
        it = self.sigmoid(self.sigmoid(self.input_layer(combine)))
        
        cell_state = cell * ft + ct_ * it
        out = self.sigmoid(self.output_layer(combine))
        hidden_state = out * self.tanh(cell_state)

        return out, hidden_state, cell_state
    
class LSTM(nn.Module): # batch == 1
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.bidirectional = bidirectional
        
    def forward(self, input, hidden, cell):
        h0 = torch.randn(self.num_layers, self.hidden_size)
        c0 = torch.randn(self.num_layers, self.hidden_size)
        
        hidden_state = []
        cell_state = []
        out = []
        
        for n in range(self.num_layers - 1):
            o_pre, h_pre, c_pre = None, None, None
            if (n == 0):
                o_n, h_n, c_n = LSTMCell(input, h0, c0)
            else:
                o_n, h_n, c_n = LSTMCell(o_pre, h_pre, c_pre)
            hidden_state.append(h_n)
            cell_state.append(c_n)
            out.append(o_n)
            o_pre, h_pre, c_pre = o_n, h_n, c_n
            
        return out, hidden_state, cell_state
            