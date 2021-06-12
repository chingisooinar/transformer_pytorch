#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:48:13 2021

@author: chingis
"""
import torch.nn as nn
from .attention import SelfAttention
 
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        Parameters
        ----------
        embed_size : int
            Embedding size.
        heads : int
            Number of heads for multi head attention.
        dropout : float
            Drop out rate.
        forward_expansion : int
            Forward expansion.
        Returns
        -------
        None.

        """
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        # Multi Head attention
        attention = self.attention(value, key, query, mask)
        # Add & Norm 
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        """
        Parameters
        ----------
        embed_size : int 
            Embedding size.
        heads : int
            Number of heads for multi head attention.
        forward_expansion : int
            Forward expansion.
        dropout : float
            dropout rate.
        device : str
            device type.

        Returns
        -------
        None.

        """
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, trg_mask, src_mask=None):
        """
        Parameters
        ----------
        x : tensor
            Decoded Embeddings.
        value : tensor
            Output of Encoder.
        key : tensor
            same output of Encoder.
        trg_mask : tensor
            Mask for subsequent words.
        src_mask : tensor, optional
            Mask for [pad]. The default is None.

        Returns
        -------
        tensor.
        """
        # Masked attention
        attention = self.attention(x, x, x, trg_mask)
        # Add & Norm 
        query = self.dropout(self.norm(attention + x))
        # Encoder - Decoder attention 
        out = self.transformer_block(value, key, query, src_mask)
        return out