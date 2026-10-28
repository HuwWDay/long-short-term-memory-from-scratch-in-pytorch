"""
Long Short-Term Memory from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rnn_forward
import torch
import torch.nn as nn

def init_rnn(d, h, seed, scale=0.5):
    """
    Initializes RNN parameters.
    
    Returns:
        dict: {'Wx': nn.Parameter (h, d), 
               'Wh': nn.Parameter (h, h), 
               'b': nn.Parameter (h,)}
    """
    torch.manual_seed(seed)
    Wx = nn.Parameter(scale * torch.randn(h, d))
    Wh = nn.Parameter(scale * torch.randn(h, h))
    b = nn.Parameter(torch.zeros(h))
    return {'Wx': Wx, 'Wh': Wh, 'b': b}

def rnn_step(x, h_prev, p):
    """
    Computes a single step of the RNN:
        h_t = tanh(x @ Wx.T + h_prev @ Wh.T + b)
        
    Args:
        x: Input tensor of shape (B, d)
        h_prev: Previous hidden state of shape (B, h)
        p: Dict containing parameters {'Wx', 'Wh', 'b'}
        
    Returns:
        torch.Tensor of shape (B, h)
    """
    return torch.tanh(x @ p['Wx'].T + h_prev @ p['Wh'].T + p['b'])

def rnn_forward(X, p):
    """
    Runs the RNN over a sequence of inputs from an initial zero state.
    
    Args:
        X: Input tensor of shape (B, T, d)
        p: Dict containing parameters {'Wx', 'Wh', 'b'}
        
    Returns:
        torch.Tensor of shape (B, T, h) containing all hidden states.
    """
    B, T, _ = X.shape
    h_dim = p['Wh'].shape[0]
    
    # Initialize initial hidden state to zeros on the same device and dtype as X
    h_t = torch.zeros(B, h_dim, device=X.device, dtype=X.dtype)
    hidden_states = []
    
    for t in range(T):
        h_t = rnn_step(X[:, t, :], h_t, p)
        hidden_states.append(h_t)
        
    return torch.stack(hidden_states, dim=1)

