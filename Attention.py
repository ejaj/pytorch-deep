import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder Class
class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.gru = nn.GRU(n_features, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden


# Attention Class
class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, proj_values=False):
        super().__init__()
        self.d_k = hidden_dim
        self.input_dim = hidden_dim if input_dim is None else input_dim
        self.proj_values = proj_values
        # Affine transformations for Q, K, and V
        self.linear_query = nn.Linear(self.input_dim, hidden_dim)
        self.linear_key = nn.Linear(self.input_dim, hidden_dim)
        self.linear_value = nn.Linear(self.input_dim, hidden_dim)
        self.alphas = None

    def init_keys(self, keys):
        self.keys = keys
        self.proj_keys = self.linear_key(self.keys)
        self.values = self.linear_value(self.keys) if self.proj_values else self.keys

    def score_function(self, query):
        proj_query = self.linear_query(query)
        # scaled dot product
        # N, 1, H x N, H, L -> N, 1, L
        dot_products = torch.bmm(proj_query, self.proj_keys.permute(0, 2, 1))
        scores = dot_products / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        return scores

    def forward(self, query, mask=None):
        # Query is batch-first N, 1, H
        scores = self.score_function(query)  # N, 1, L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1)  # N, 1, L
        self.alphas = alphas.detach()

        # N, 1, L x N, L, H -> N, 1, H
        context = torch.bmm(alphas, self.values)
        return context


# Decoder with Attention Class
class DecoderAttn(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(DecoderAttn, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.attn = Attention(self.hidden_dim)
        self.regression = nn.Linear(2 * self.hidden_dim, self.n_features)

    def init_hidden(self, hidden_seq):
        # the output of the encoder is N, L, H
        # and init_keys expects batch-first as well
        self.attn.init_keys(hidden_seq)
        hidden_final = hidden_seq[:, -1:]
        self.hidden = hidden_final.permute(1, 0, 2)  # L, N, H

    def forward(self, X, mask=None):
        # X is N, 1, F
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        query = batch_first_output[:, -1:]

        # Attention
        context = self.attn(query, mask=mask)
        concatenated = torch.cat([context, query], axis=-1)

        out = self.regression(concatenated)
        # N, 1, F
        return out.view(-1, 1, self.n_features)


# Encoder-Decoder with Attention Class
class EncoderDecoderAttn(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.alphas = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # N, L (target), F
        self.outputs = torch.zeros(batch_size, self.target_len, self.encoder.n_features).to(device)
        # N, L (target), L (source)
        self.alphas = torch.zeros(batch_size, self.target_len, self.input_len).to(device)

    def store_output(self, i, out):
        # Stores the output
        self.outputs[:, i:i + 1, :] = out
        self.alphas[:, i:i + 1, :] = self.decoder.attn.alphas

    def forward(self, X, target=None):
        batch_size = X.size(0)
        self.init_outputs(batch_size)
        encoder_outputs, hidden = self.encoder(X)

        self.decoder.init_hidden(encoder_outputs)
        decoder_input = X[:, -1:, :]  # Initial input to the decoder

        for i in range(self.target_len):
            out = self.decoder(decoder_input)
            self.store_output(i, out)

            if target is not None and torch.rand(1).item() < self.teacher_forcing_prob:
                decoder_input = target[:, i:i + 1, :]
            else:
                decoder_input = out

        return self.outputs


# Example usage with sample data
def main():
    # Define the input sequence
    full_seq = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float32).view(1, 4, 2)
    source_seq = full_seq[:, :2]  # First two corners (input)
    target_seq = full_seq[:, 2:]  # Last two corners (output)

    # Initialize the encoder and decoder
    input_dim = source_seq.size(2)
    hidden_dim = 2
    encoder = Encoder(input_dim, hidden_dim)
    decoder = DecoderAttn(input_dim, hidden_dim)

    # Combine them into the EncoderDecoder model
    model = EncoderDecoderAttn(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)

    # Run the model
    model.train()
    output_train = model(source_seq, target=target_seq)
    print("Training Output:")
    print(output_train)

    # Switch to evaluation mode
    model.eval()
    output_eval = model(source_seq)
    print("\nEvaluation Output:")
    print(output_eval)

    # Print attention weights
    print("\nAttention Weights:")
    print(model.alphas)


if __name__ == "__main__":
    main()
