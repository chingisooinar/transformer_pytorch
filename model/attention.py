#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:46:11 2021

@author: chingis
"""
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        Parameters
        ----------
        embed_size : int
            Embedding size.
        heads : int
            Number of heads in multi head attention.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.sqrt_d = math.sqrt(self.embed_size)
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be div by heads"
        
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        #Embedding size needs to be div by heads
        self.fc_out = nn.Linear(self.head_dim * heads, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        keys = self.K(keys)
        values = self.V(values)
        queries = self.Q(queries)
        
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
            
        # Softmax
        attention = torch.softmax(energy / self.sqrt_d, dim=-1)  #normalize across key_len
        # out shape: (N, query_len, heads, head_dim)
        # key_len = value_len -> l
        out = torch.einsum('nhql, nlhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim    
        )  # concatenation -> shape: N, query_len, embed_size
        
        return self.fc_out(out)
    