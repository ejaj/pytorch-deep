import torch
import torch.nn.functional as F

# Given
query = torch.tensor([[0.5, 1.0]])
keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
values = keys  # For simplicity, using keys as values
print(keys.t())
# Step 1: Calculate similarity scores
scores = torch.mm(query, keys.t())  # (1, 2) x (2, 3) -> (1, 3)
# Step 2: Apply softmax to get attention weights
attn_weights = F.softmax(scores, dim=-1)  # (1, 3)

# Step 3: Compute context vector
context = torch.mm(attn_weights, values)  # (1, 3) x (3, 2) -> (1, 2)
print("Scores:", scores)
print("Attention Weights:", attn_weights)
print("Context Vector:", context)
