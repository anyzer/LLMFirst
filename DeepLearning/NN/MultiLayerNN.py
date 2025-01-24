import torch
import torch.nn.functional as F
from torch.autograd import grad

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 3000),
            torch.nn.ReLU(),

            torch.nn.Linear(3000, 2000),
            torch.nn.ReLU(),

            torch.nn.Linear(2000, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)

print(model.layers[0].bias)

torch.manual_seed(123)

X = torch.rand((1, 50))
with torch.no_grad():
    out = model(X)
print(out)


