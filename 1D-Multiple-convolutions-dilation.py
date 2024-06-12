import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

## 1D Convolutions
temperatures = np.array([5, 11, 15, 6, 5, 3, 3, 0, 0, 3, 4, 2, 1])
size = 5
weight = torch.ones(size) * 0.2  # A simple moving average filter

# Perform the convolution
output = F.conv1d(torch.as_tensor(temperatures).float().view(1, 1, -1),
                  weight=weight.view(1, 1, -1))
print(output)

points = np.random.rand(128, 4, 2)  # Example sequences of shape (batch_size, seq_length, features)
seqs = torch.as_tensor(points).float()  # N, L, F
seqs_length_last = seqs.permute(0, 2, 1)  # Convert to N, F, L
print(seqs_length_last.shape)

## Multiple Features or Channels
torch.manual_seed(17)
conv_seq = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, bias=False)
print(conv_seq.weight, conv_seq.weight.shape)

# Convolve the sequence
output = conv_seq(seqs_length_last[0:1])
print(output)

## Dilation
torch.manual_seed(17)
conv_dilated = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, dilation=2, bias=False)
print(conv_dilated.weight, conv_dilated.weight.shape)

# Convolve the sequence with dilation
output_dilated = conv_dilated(seqs_length_last[0:1])
print(output_dilated)
