import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, input_dim=None, proj_values=True):
        super().__init__()
        self.d_model = d_model
        self.proj_values = proj_values

        self.query = nn.Linear(input_dim, d_model) if input_dim else nn.Linear(d_model, d_model)
        self.key = nn.Linear(input_dim, d_model) if input_dim else nn.Linear(d_model, d_model)
        self.value = nn.Linear(input_dim, d_model) if input_dim else nn.Linear(d_model, d_model)

        if proj_values:
            self.proj_value = nn.Linear(d_model, d_model)
        else:
            self.proj_value = None

        self.alphas = None

    def init_keys(self, key):
        self.keys = self.key(key)

    def forward(self, query, mask=None):
        Q = self.query(query)
        K = self.keys
        V = self.value(query)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model).float())

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        self.alphas = F.softmax(scores, dim=-1)
        context = torch.matmul(self.alphas, V)

        if self.proj_value is not None:
            context = self.proj_value(context)

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, input_dim=None, proj_values=True):
        super().__init__()
        self.linear_out = nn.Linear(n_heads * d_model, d_model)
        self.attn_heads = nn.ModuleList(
            [Attention(d_model, input_dim=input_dim, proj_values=proj_values)
             for _ in range(n_heads)]
        )

    def init_keys(self, key):
        for attn in self.attn_heads:
            attn.init_keys(key)

    @property
    def alphas(self):
        # Shape: n_heads, N, 1, L (source)
        return torch.stack([attn.alphas for attn in self.attn_heads], dim=0)

    def output_function(self, contexts):
        # N, 1, n_heads * D
        concatenated = torch.cat(contexts, axis=-1)
        # Linear transformation to go back to original dimension
        out = self.linear_out(concatenated)  # N, 1, D
        return out

    def forward(self, query, mask=None):
        contexts = [attn(query, mask=mask) for attn in self.attn_heads]
        out = self.output_function(contexts)
        return out


# Example usage
n_heads = 8
d_model = 64
input_dim = 512

multi_head_attn = MultiHeadAttention(n_heads, d_model, input_dim)
query = torch.randn(1, 10, input_dim)
key = torch.randn(1, 10, input_dim)
mask = None

multi_head_attn.init_keys(key)
output = multi_head_attn(query, mask)
print(output.shape)  # Expected shape: (1, 10, d_model)
