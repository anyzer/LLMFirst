import torch
import torch.nn.functional as F
from torch.autograd import grad

print("Hello, World")

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])

w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)
# grad_L__w1 = grad(loss, w1, retain_graph=True)
# gradL__b = grad(loss, b, retain_graph=True)
loss.backward()
print(w1.grad)
print(b.grad)

