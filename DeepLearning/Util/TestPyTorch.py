import torch
import torch.nn.functional as F
from torch.autograd import grad

print("Hello, World")

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

tensor_3 = torch.tensor([1, 2, 3])

tensor_4 = torch.tensor([[1, 2, 3]])
vertical_tensor = torch.tensor([[1], [2], [3]])

print(tensor_4)
print(vertical_tensor)

print(tensor_4 * vertical_tensor)

# print(tensor_4 @ vertical_tensor)
# print(" ")
# print(vertical_tensor @ tensor_4)