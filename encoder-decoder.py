import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(input_size=self.n_features, hidden_size=self.hidden_dim, batch_first=True)

    def forward(self, x):
        rnn_out, self.hidden = self.basic_rnn(x)
        return rnn_out  # N, L, F


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.regression = nn.Linear(self.hidden_dim, self.n_features)

    def init_hidden(self, hidden_seq):
        hidden_final = hidden_seq[:, -1:]  # N, 1, H
        self.hidden = hidden_final.permute(1, 0, 2)  # 1, N, H

    def forward(self, X):
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        last_output = batch_first_output[:, -1:]
        out = self.regression(last_output)
        return out.view(-1, 1, self.n_features)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.outputs = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        self.outputs = torch.zeros(batch_size, self.target_len, self.encoder.n_features).to(device)

    def store_output(self, i, out):
        self.outputs[:, i:i + 1, :] = out

    def forward(self, x):
        source_seq = x[:, :self.input_len, :]
        target_seq = x[:, self.input_len:, :]
        self.init_outputs(x.shape[0])
        hidden_seq = self.encoder(source_seq)
        self.decoder.init_hidden(hidden_seq)
        dec_inputs = source_seq[:, -1:, :]

        for i in range(self.target_len):
            out = self.decoder(dec_inputs)
            self.store_output(i, out)
            prob = self.teacher_forcing_prob if self.training else 0
            if torch.rand(1) <= prob:
                dec_inputs = target_seq[:, i:i + 1, :]
            else:
                dec_inputs = out
        return self.outputs


full_seq = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float32).view(1, 4, 2)
source_seq = full_seq[:, :2]  # first two corners
target_seq = full_seq[:, 2:]  # last two corners

# Initialize the encoder and decoder
torch.manual_seed(21)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder = Decoder(n_features=2, hidden_dim=2)

# Combine them into the EncoderDecoder model
encdec = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)

# Training mode
encdec.train()
output_train = encdec(full_seq)
print("Training Output:")
print(output_train)

# Evaluation mode
encdec.eval()
output_eval = encdec(source_seq)
print("\nEvaluation Output:")
print(output_eval)
