import torch
import torch.nn as nn
import torch.nn.functional as F


class NarrowAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(NarrowAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, depth)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project the inputs to the dimension size of d_model
        query = self.split_heads(self.linear_query(query))
        key = self.split_heads(self.linear_key(key))
        value = self.split_heads(self.linear_value(value))

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        output = self.linear_out(context)
        return output


n_heads = 8
d_model = 512
input_data = torch.randn(1, 10, d_model)  # (batch_size, seq_len, d_model)

narrow_attention = NarrowAttention(n_heads, d_model)
output = narrow_attention(input_data, input_data, input_data)
print("Narrow Attention Output Shape:", output.shape)
