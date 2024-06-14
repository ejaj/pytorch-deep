import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.depth, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        scaled_attention = torch.matmul(attention_weights, v)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(original_size_attention)
        return output


# Define the Encoder with Self-Attention
class EncoderSelfAttn(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super(EncoderSelfAttn, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = n_features
        self.self_attn_heads = MultiHeadAttention(n_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model)
        )

    def forward(self, query, mask=None):
        att = self.self_attn_heads(query, query, query, mask)
        out = self.ffn(att)
        return out


# Define the Decoder with Self-Attention and Cross-Attention
class DecoderSelfAttn(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super(DecoderSelfAttn, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = d_model if n_features is None else n_features

        self.self_attn_heads = MultiHeadAttention(n_heads, d_model)
        self.cross_attn_heads = MultiHeadAttention(n_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, self.n_features)
        )

    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        att1 = self.self_attn_heads(query, query, query, target_mask)
        att2 = self.cross_attn_heads(att1, att1, att1, source_mask)
        out = self.ffn(att2)
        return out


# Define the Encoder-Decoder Model with Cross-Attention
class EncoderDecoderSelfAttn(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len):
        super(EncoderDecoderSelfAttn, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.trg_masks = self.subsequent_mask(self.target_len)

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = (1 - torch.triu(torch.ones(attn_shape), diagonal=1)).bool()
        return subsequent_mask

    def encode(self, source_seq, source_mask):
        encoder_states = self.encoder(source_seq, source_mask)
        self.decoder.init_keys(encoder_states)

    def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
        outputs = self.decoder(shifted_target_seq, source_mask=source_mask, target_mask=target_mask)
        return outputs

    def predict(self, source_seq, source_mask):
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.decode(inputs, source_mask, self.trg_masks[:, :i + 1, :i + 1])
            out = torch.cat([inputs, out[:, -1:, :]], dim=-2)
            inputs = out.detach()
        outputs = inputs[:, 1:, :]
        return outputs

    def forward(self, X, source_mask=None):
        self.trg_masks = self.trg_masks.type_as(X).bool()
        source_seq = X[:, :self.input_len, :]
        self.encode(source_seq, source_mask)
        if self.training:
            shifted_target_seq = X[:, self.input_len - 1:-1, :]
            outputs = self.decode(shifted_target_seq, source_mask, self.trg_masks)
        else:
            outputs = self.predict(source_seq, source_mask)
        return outputs


# Example usage
torch.manual_seed(11)
n_heads = 3
d_model = 6
ff_units = 10
n_features = 6

# Instantiate the encoder and decoder
encoder = EncoderSelfAttn(n_heads, d_model, ff_units, n_features)
decoder = DecoderSelfAttn(n_heads, d_model, ff_units, n_features)

# Instantiate the EncoderDecoderSelfAttn class
encoder_decoder = EncoderDecoderSelfAttn(encoder, decoder, input_len=4, target_len=2)

# Define a small input sequence (batch_size=1, seq_len=4, d_model=6)
source_seq = torch.randn(1, 4, n_features)
target_seq = torch.randn(1, 2, n_features)
X = torch.cat([source_seq, target_seq], dim=1)

# Apply the encoder-decoder model
encoder_decoder.train()
output = encoder_decoder(X)
print("Output Sequence:\n", output)
