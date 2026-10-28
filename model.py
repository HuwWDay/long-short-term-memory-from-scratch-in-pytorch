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

# Step 2 - gradient_vs_lag
import math
import numpy as np
import torch

def gradient_vs_lag(forward, X, lags):
    """
    Computes L2 gradient norms of the input at time step T - 1 - lag with respect
    to the sum of the last time step's hidden states.
    
    Args:
        forward: Callable mapping X of shape (B, T, d) to states of shape (B, T, h).
        X: Input tensor of shape (B, T, d).
        lags: Iterable of non-negative integer lags.
        
    Returns:
        List of float L2 norms of the gradient for each lag.
    """
    X_input = X.clone().detach().requires_grad_(True)
    states = forward(X_input)  # Shape: (B, T, h)
    
    T = states.shape[1]
    # Loss is the sum of the last step's states
    loss = states[:, -1, :].sum()
    loss.backward()
    
    # X_input.grad has shape (B, T, d)
    grad = X_input.grad
    norms = []
    for lag in lags:
        step_idx = T - 1 - lag
        # Compute Frobenius / L2 norm over (B, d) at the target time step
        step_norm = torch.linalg.norm(grad[:, step_idx, :]).item()
        norms.append(float(step_norm))
        
    return norms

def decay_rate(lags, norms):
    """
    Computes the least-squares slope of log(norms) against lags.
    
    Args:
        lags: Array-like of integer lags.
        norms: Array-like of gradient norms.
        
    Returns:
        float: The slope of the least-squares fit line.
    """
    x = np.asarray(lags, dtype=np.float64)
    y = np.log(np.asarray(norms, dtype=np.float64))
    
    # Fit degree 1 polynomial: y = slope * x + intercept
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)

def half_life(rate):
    """
    Computes the half-life from the decay rate: ln(0.5) / rate for rate < 0,
    otherwise returns float('inf').
    
    Args:
        rate: float decay rate (slope).
        
    Returns:
        float: Half-life in time steps, or float('inf').
    """
    if rate < 0:
        return math.log(0.5) / rate
    return float('inf')

