import torch
import torch.nn.utils.rnn as rnn_utils

seq1 = [[1.0349, 0.9661], [0.8055, -0.9169], [-0.8251, -0.9499], [-0.8670, 0.9342]]
seq2 = [[-1.0911, 0.9254], [-1.0771, -1.0414]]
seq3 = [[-1.1247, -0.9683], [0.8182, -0.9944], [1.0081, 0.7680]]

all_tensors = [seq1, seq2, seq3]

seq_tensors = [torch.tensor(seq).float() for seq in all_tensors]
padded = rnn_utils.pad_sequence(seq_tensors, batch_first=True)
print(padded)
packed = rnn_utils.pack_sequence(seq_tensors, enforce_sorted=False)
print(packed)

rnn = torch.nn.RNN(2, 2, batch_first=True)
output_packed, hidden_packed = rnn(packed)

# Unpack the sequences
output_unpacked, seq_sizes = rnn_utils.pad_packed_sequence(output_packed, batch_first=True)
print(output_unpacked)
print(seq_sizes)
