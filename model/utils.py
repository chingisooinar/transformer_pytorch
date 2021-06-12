#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:47:29 2021

@author: chingis
"""
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, max_len = 80):
        super().__init__()
        self.embed_size = embed_size
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/embed_size)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_size)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x