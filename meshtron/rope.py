import torch

def precompute_theta_pos_freq(head_dim, seq_len, theta):
  assert head_dim % 2 == 0
  theta_numerator = torch.arange(0, head_dim, 2).float()
  theta = 1/ (theta ** (theta_numerator/head_dim))
  m = torch.arange(seq_len)
  freqs = torch.outer(m, theta).float()
  freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_complex

def apply_rope(x, freqs_complex):
  x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
  freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
  x_rotated = x_complex*freqs_complex
  x_out = torch.view_as_real(x_rotated)
  x_out = x_out.reshape(*x.shape)
  return x_out.type_as(x)