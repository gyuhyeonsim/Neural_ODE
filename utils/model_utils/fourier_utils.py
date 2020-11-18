import torch

def FourierExpansion(n_range, s):
    """Fourier eigenbasis expansion
    """
    s_n_range = s*n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis

def PolyExpansion(n_range, s):
    """Polynomial expansion
    """
    basis = [s**n_range]
    return basis



