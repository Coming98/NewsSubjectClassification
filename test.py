from torch.serialization import load
from torch.utils.data import TensorDataset, DataLoader
import torch

X = torch.tensor([ [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7] ])
y = torch.tensor([-1, 1, 1, -1, 1])
dataset = TensorDataset(X, y)

def generate_batch(batch_dataset):
    input = []
    label = []
    for demo_input, demo_label in batch_dataset:
            input.append(list(demo_input))
            label.append(0 if demo_label == -1 else demo_label.item())
    return torch.tensor(input), torch.tensor(label)

loader_customised = DataLoader(dataset, batch_size=2, drop_last=True, shuffle=True, collate_fn=generate_batch)

for i, (input, label) in enumerate(loader_customised):
    print(i)
    print(input)
    print()
    print(label)
    