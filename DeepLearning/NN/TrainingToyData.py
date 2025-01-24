import torch
import torch.nn.functional as F
from MultiLayerNN import NeuralNetwork
from ToyDataset import train_loader, X_train, y_train, test_loader
from datetime import datetime

# python DeepLearning/NN/TrainingToyData.py

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()
    
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("mps")
# device = torch.device("cpu")
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        features, labels = features.to(device), labels.to(device) 
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad() # reset grad
        loss.backward() # calculate loss
        optimizer.step() # update model parameters
        
        current_timestamp = datetime.now()
        print("Current:", current_timestamp)

        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
        

model.eval()

print("MPS Count:", torch.mps.device_count())
print("CPU Count:", torch.cpu.device_count())

# with torch.no_grad():
#     outputs = model(X_train)
# print(outputs)

# torch.set_printoptions(sci_mode=False)
# probas = torch.softmax(outputs, dim=1)
# print(probas)

# predictions = torch.argmax(probas, dim=1)
# print(predictions)

# predictions1 = torch.argmax(outputs, dim=1)
# print(predictions1)

# print(predictions1 == y_train)

# print(torch.sum(predictions == y_train))
# # count the number of correct prediction

# print("Compute Accuracy")
# print(compute_accuracy(model, train_loader))
# print(compute_accuracy(model, test_loader))

# torch.save(model.state_dict(), "./DeepLearning/Save/FirstNN.pth")

# model = NeuralNetwork(2, 2)
# model.load_state_dict(torch.load("model.pth"))