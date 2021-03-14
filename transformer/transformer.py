"""
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import torch
import torch.nn as nn

class CasualSelfAttention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 block_size,
                 num_layers,
                 num_heads,
                 embedding_dropout=0.1,
                 redisudal_dropout=0.1,
                 attention_dropout=0.1,
                 **kwargs):
        super(CasualSelfAttention, self).__init__()
        
        # key, query, value
        self.key = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # regularization
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(redisudal_dropout)
        
        # out projection
        self.projection = nn.Linear(embed_size, embed_size)
        
        #casual mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        
        self.num_heads = num_heads

    def forward(self, xx, layer_past=None):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C// self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        att = self.attention_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.residual_dropout(self.proj(y))
        
        return y      