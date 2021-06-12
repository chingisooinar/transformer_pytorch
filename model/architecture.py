#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:49:55 2021

@author: chingis
"""
import torch.nn as nn
from .utils import PositionalEncoder
from .transformer_block import TransformerBlock, DecoderBlock

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layer,
                 heads, device, forward_expansion, dropout,
                 max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoder(embed_size, max_len)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layer)
            ]    
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        """
        Parameters
        ----------
        x : tensor
            sentences.
        mask : tensor
            Mask for [pad].

        Returns
        -------
        out : tensor
        """
        N, seq_length = x.shape
        
        # Embedding
        x = self.word_embedding(x)  # Shape: N, seq_length, embed_size  
        x = self.position_embedding(x)
        out = self.dropout(x)
        # Pass trhough TransformerBlocks
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    

class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size, embed_size, num_layers, heads, forward_expansion,
                 dropout, device, max_len):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoder(embed_size, max_len)#nn.Embedding(max_len, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]    
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        Parameters
        ----------
        x : tensor
            target words.
        enc_out : tensor
            Encoder output.
        src_mask : tensor
            Mask for [pad].
        trg_mask : tensor
            Mask for subsequent target words.

        Returns
        -------
        out : tensor
            next words.
        """
        N, seq_length = x.shape

        # Embedding       
        x = self.word_embedding(x)  # Shape: N, seq_length, embed_size
        x = self.position_embedding(x)
        x = self.dropout(x)
        # Pass through DecoderBlocks
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, trg_mask, src_mask)
            
        out = self.fc_out(x)
        return out