import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def make_chunks(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        x = x.transpose(1, 2)
        return x

    def init_keys(self, key):
        self.proj_key = self.make_chunks(self.linear_key(key))
        self.proj_value = self.make_chunks(self.linear_value(key))

    def score_function(self, query):
        proj_query = self.make_chunks(self.linear_query(query))
        scores = torch.matmul(proj_query, self.proj_key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        return scores

    def attn(self, query, mask=None):
        scores = self.score_function(query)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1)
        alphas = self.dropout(alphas)
        context = torch.matmul(alphas, self.proj_value)
        return context

    def forward(self, query, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        context = self.attn(query, mask=mask)
        context = context.transpose(1, 2).contiguous()
        context = context.view(query.size(0), -1, self.d_model)
        out = self.linear_out(context)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed)
        pe[:, 1::2] = torch.cos(position * angular_speed)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        scaled_x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, query, mask=None):
        norm_query = self.norm1(query)
        self.self_attn_heads.init_keys(norm_query)
        states = self.self_attn_heads(norm_query, mask)
        att = query + self.drop1(states)
        norm_att = self.norm2(att)
        out = self.ffn(norm_att)
        out = att + self.drop2(out)
        return out


class EncoderTransformer(nn.Module):
    def __init__(self, encoder_layer, n_layers=1, max_len=100):
        super(EncoderTransformer, self).__init__()
        self.d_model = encoder_layer.d_model
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, query, mask=None):
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.cross_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.d_model = d_model

    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        norm_query = self.norm1(query)
        self.self_attn_heads.init_keys(norm_query)
        states = self.self_attn_heads(norm_query, target_mask)
        att1 = query + self.drop1(states)
        norm_att1 = self.norm2(att1)
        encoder_states = self.cross_attn_heads(norm_att1, source_mask)
        att2 = att1 + self.drop2(encoder_states)
        norm_att2 = self.norm3(att2)
        out = self.ffn(norm_att2)
        out = att2 + self.drop3(out)
        return out


class DecoderTransformer(nn.Module):
    def __init__(self, decoder_layer, n_layers=1, max_len=100):
        super(DecoderTransformer, self).__init__()
        self.d_model = decoder_layer.d_model
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])

    def init_keys(self, states):
        for layer in self.layers:
            layer.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, source_mask, target_mask)
        return self.norm(x)


# Define parameters
n_heads = 8
d_model = 512
ff_units = 2048
dropout = 0.1
n_layers = 6
max_len = 100

# Initialize encoder and decoder
encoder_layer = EncoderLayer(n_heads, d_model, ff_units, dropout)
encoder = EncoderTransformer(encoder_layer, n_layers, max_len)

decoder_layer = DecoderLayer(n_heads, d_model, ff_units, dropout)
decoder = DecoderTransformer(decoder_layer, n_layers, max_len)

# Example input sequences
source_seq = torch.randn(1, 10, d_model)  # (batch_size, seq_len, d_model)
target_seq = torch.randn(1, 10, d_model)  # (batch_size, seq_len, d_model)

# Pass through encoder and decoder
encoder_output = encoder(source_seq)
decoder.init_keys(encoder_output)
decoder_output = decoder(target_seq)

print("Encoder Output Shape:", encoder_output.shape)
print("Decoder Output Shape:", decoder_output.shape)
