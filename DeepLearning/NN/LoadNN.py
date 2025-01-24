import torch
import torch.nn.functional as F
from MultiLayerNN import NeuralNetwork
from ToyDataset import train_loader, X_train, y_train, test_loader
from TrainingToyData import compute_accuracy
    
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
model.load_state_dict(torch.load("./DeepLearning/Save/FirstNN.pth"))

print(model.layers[0].weight)
print(model.layers[0].weight.shape)
print(compute_accuracy(model, test_loader))

print(torch.__version__) 
print(torch.backends.mps.is_available())  # True if MPS backend is available
print(torch.backends.mps.is_built())  # True if PyTorch was built with MPS support

