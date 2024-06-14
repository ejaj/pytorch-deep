import torch
import torch.nn as nn


class WideAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(WideAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_heads = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_heads)])
        self.linear_out = nn.Linear(n_heads * d_model, d_model)

    def forward(self, query, key, value):
        # Each attention head processes the full input
        attn_outputs = [head(query) for head in self.attn_heads]
        # Concatenate the ouputs
        concatenated = torch.cat(attn_outputs, dim=-1)
        # Pass through a final linear layer
        output = self.linear_out(concatenated)
        return output


n_heads = 8
d_model = 512
input_data = torch.randn(1, 10, d_model)  # (batch_size, seq_len, d_model)

wide_attention = WideAttention(n_heads, d_model)
output = wide_attention(input_data, input_data, input_data)
print("Wide Attention Output Shape:", output.shape)
