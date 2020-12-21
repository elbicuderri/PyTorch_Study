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


class LSTMCell_V2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_V2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.hi_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hf_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hg_layer = nn.Linear(self.input_size, self.hidden_size)
        self.ho_layer = nn.Linear(self.input_size, self.hidden_size)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, hidden):
                
        i = self.sigmoid(self.hi_layer(hidden))
        # f = self.sigmoid(self.hf_layer(hidden))
        g = self.tanh(self.hg_layer(hidden))
        out = self.sigmoid(self.ho_layer(hidden))               
        
        cell_state = i * g # hadamard product Not matmul
        hidden_state = out*self.tanh(cell_state)
        
        return out, hidden_state, cell_state
    
class LSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.bidirectional = bidirectional
        
        self.lstm_cell_v0 = LSTMCell_V2(self.input_size, self.hidden_size)
        self.lstm_cell_v1 = LSTMCell(self.input_size, self.hidden_size)
        self.lstm_cell_v2 = LSTMCell_V2(self.hidden_size, self.hidden_size)
        self.lstm_cell_v3 = LSTMCell(self.hidden_size, self.hidden_size)
        
    def forward(self, input):
        # input : (seq_len, batch, input_size)
        # h_0 : (num_layers, batch, hidden_size) if bi (num_layers * 2, batch, hidden_size)
        # c_0 : (num_layers, batch, hidden_size) if bi (num_layers * 2, batch, hidden_size)
                
        seq_len = input.size(0)  
        batch = input.size(1)
                
        out_list = torch.zeros(seq_len, batch, self.hidden_size) # (seq_len, batch, hidden_size)
        # hidden_states = torch.zeros(self.num_layers, batch, self.hidden_size) # (num_layers, batch, hidden_size)
        # cell_states = torch.zeros(self.num_layers, batch, self.hidden_size) # (num_layers, batch, hidden_size)

        for l in range(seq_len):
            # out_list = torch.zeros(seq_len, batch, self.hidden_size)

            o_previous = torch.zeros(1, batch, self.hidden_size)
            h_previous = torch.zeros(1, batch, self.hidden_size)
            c_previous = torch.zeros(1, batch, self.hidden_size)

            h_previous_list = torch.zeros(self.num_layers, batch, self.hidden_size) # (t-1) hidden states list
            c_previous_list = torch.zeros(self.num_layers, batch, self.hidden_size) # (t-1) cell states list

            if (l == 0):
                output = 0.0
                for layer_num in range(self.num_layers):
                    if (layer_num == 0):
                        o_previous, h_previous, c_previous = self.lstm_cell_v0(input[0,:,:])
                        h_previous_list[0,:,:] = h_previous # hidden states list update
                        c_previous_list[0,:,:] = c_previous # cell states list update
                    else:
                        o_previous, h_previous, c_previous = self.lstm_cell_v2(o_previous) 
                        h_previous_list[layer_num,:,:] = h_previous # hidden states list update
                        c_previous_list[layer_num,:,:] = c_previous # cell states list update
                    output = o_previous # last output
                out_list[0,:,:] = output

            else:
                output = 0.0
                for layer_num in range(self.num_layers):
                    if (layer_num == 0):
                        o_previous, h_previous, c_previous = self.lstm_cell_v1(input[l,:,:], h_previous_list[0,:,:], c_previous_list[0,:,:])
                        h_previous_list[0,:,:] = h_previous # hidden states list update
                        c_previous_list[0,:,:] = c_previous # cell states list update

                    else:
                        o_previous, h_previous, c_previous = self.lstm_cell_v3(h_previous, h_previous_list[layer_num,:,:], c_previous_list[layer_num,:,:])
                        h_previous_list[layer_num,:,:] = h_previous # hidden states list update
                        c_previous_list[layer_num,:,:] = c_previous # cell states list update
                    output = o_previous # last output
                out_list[l,:,:] = output # l
                                             
        return out_list
    

seq_len = 100
batch = 16
input_size = 16
hidden_size = 32
num_layers = 16

InTensor = torch.randn(seq_len, batch, input_size)

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

out = model(InTensor) # (seq_len, batch, hidden_size)

print(out.size())
