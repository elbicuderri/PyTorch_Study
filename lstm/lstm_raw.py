import torch
import torch.nn as nn 

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.ii_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hi_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.if_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hf_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.ig_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hg_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.io_layer = nn.Linear(self.input_size, self.hidden_size)
        self.ho_layer = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input, hidden, cell):
                
        i = self.sigmoid(self.ii_layer(input) + self.hi_layer(hidden))
        f = self.sigmoid(self.if_layer(input) + self.hf_layer(hidden))
        g = self.tanh(self.ig_layer(input) + self.hg_layer(hidden))
        out = self.sigmoid(self.io_layer(input) + self.ho_layer(hidden))               
        
        cell_state = (f * cell) + (i * g) # hadamard product Not matmul
        hidden_state = out*self.tanh(cell_state)
        
        return out, hidden_state, cell_state
    
class LSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.bidirectional = bidirectional
        
        self.lstm_cell = LSTMCell(self.input_size, self.hidden_size)
        
    def forward(self, input):
        # input : (seq_len, batch, input_size)
        # h_0 : (num_layers, batch, hidden_size) if bi (num_layers * 2, batch, hidden_size)
        # c_0 : (num_layers, batch, hidden_size) if bi (num_layers * 2, batch, hidden_size)
                
        seq_len = input.size(0)  
        batch = input.size(1)
                
        outs = torch.zeros(seq_len, batch, self.hidden_size) # (seq_len, batch, hidden_size)
        # hidden_states = torch.zeros(self.num_layers, batch, self.hidden_size) # (num_layers, batch, hidden_size)
        # cell_states = torch.zeros(self.num_layers, batch, self.hidden_size) # (num_layers, batch, hidden_size)

        for l in range(seq_len):
            # h_pre, c_pre = None, None
            out_sum = torch.zeros(1, batch, self.hidden_size)
            for n in range(self.num_layers):
                if (n == 0):
                    h_pre = torch.zeros(1, batch, self.hidden_size)
                    c_pre = torch.zeros(1, batch, self.hidden_size)
                    # hidden_states[:, :, :] += h_pre
                    # cell_states[:, :, :] += c_pre
                    o_n, h_n, c_n = self.lstm_cell(input[l,:,:], h_pre, c_pre)
                    assert (o_n.size() == (1, batch, self.hidden_size) and 
                            h_n.size() == (1, batch, self.hidden_size) and
                            c_n.size() == (1, batch, self.hidden_size))
                    
                else:
                    o_n, h_n, c_n = self.lstm_cell(input[l,:,:], h_pre, c_pre) # output (1, batch, hidden_size)
                    assert (o_n.size() == (1, batch, self.hidden_size) and 
                            h_n.size() == (1, batch, self.hidden_size) and
                            c_n.size() == (1, batch, self.hidden_size))
                    # outs[:, :, :] += o_n # (1, batch, hidden_size)
                    
                    # hidden_states[:, :, :] += h_n # (1, batch, hidden_size)
                    # cell_states[:, :, :] += c_n # (1, batch, hidden_size)
                    
                    h_pre, c_pre = h_n, c_n
            
                out_sum += o_n 
                
            outs[l, :, :] = out_sum
                                                     
        return outs
    

seq_len = 100
batch = 16
input_size = 16
hidden_size = 32
num_layers = 16

InTensor = torch.randn(seq_len, batch, input_size)

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

out = model(InTensor)

print(out.size())
    