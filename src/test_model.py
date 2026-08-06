import torch

from model import BrainTumorCNN

model = BrainTumorCNN()

dummy = torch.randn(1, 3, 128, 128)

output = model(dummy)

print(output.shape)
print(output)