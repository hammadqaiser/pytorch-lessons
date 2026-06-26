import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# create a 2D list and convert it to a tensor
data = [[1,2,3],[4,5,6]] # 2D list
x_data = torch.tensor(data) # tensor with the same values as data
x_rand = torch.rand_like(x_data, dtype=torch.float) # random tensor with the same shape as x_data

# define a simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    # define the forward pass of the model
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# create an instance of the model and test it with a random input tensor
model = NeuralNetwork()
X = torch.rand(1, 28, 28, 1) # random input tensor with shape (1, 28, 28, 1)
logits = model(X)
predictions = nn.Softmax(dim=1)(logits) # apply softmax to the logits to get probabilities
y_pred = predictions.argmax(1) # get the index of the maximum probability as the predicted class
print(f"Predicted class: {y_pred.item()}") # print the predicted class 
