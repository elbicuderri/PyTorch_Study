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
        # input : (seq_len, batch, input_size)
        # h_0 : (num_layer, batch, hidden_size) if bi (num_layer * 2, batch, hidden_size)
        # c_0 : (num_layer, batch, hidden_size) if bi (num_layer * 2, batch, hidden_size)
                
        seq_len = input.size(0)  
        batch = input.size(1)
                
        outs = []
        hidden_states = []
        cell_states = []
        for b in range(batch):
            b_out = 0
            b_hidden_state = []
            b_cell_state = []
            for l in range(seq_len):
                hidden_state = [] # len = n
                cell_state = [] # len = n
                out = torch.zeros(1, self.num_layers, self.hidden_size)
                for n in range(self.num_layers - 1):
                    if (l == 0):
                        h_pre = torch.zeros(self.num_layers, 1, self.hidden_size)
                        c_pre = torch.zeros(self.num_layers, 1, self.hidden_size)
                        hidden_state.append(h_pre.squeeze())
                        cell_state.append(c_pre.squeeze())
                        o_n, h_n, c_n = LSTMCell(input[l,b,:], h_pre[n,b,:], c_pre[n,b,:])
                    else:
                        o_n, h_n, c_n = LSTMCell(input[l,b,:], h_pre[n,b,:], c_pre[n,b,:]) # output (1, 1, hidden_size)
                        
                    h_pre, c_pre = h_n, c_n
                    hidden_state.append(h_n.squeeze())
                    cell_state.append(c_n.squeeze())
                    out[:, n, :] += o_n.squeeze() # (1, num_layers, hidden_size)
                    
                b_hidden_state = hidden_state
                b_cell_state = cell_state
                b_out = out # (1, seq_len, hidden_size)
                
            hidden_states.append(b_hidden_state)
            cell_states.append(b_cell_state)
            outs.append(b_out) 
            
            hidden_states = hidden_states.view(self.num_layers, batch, self.hidden_size)
            cell_states = cell_states.view(self.num_layers, batch, self.hidden_size)
            
            outs = outs.view(seq_len, batch, self.input_size)
            
        return outs, hidden_states, cell_states
            