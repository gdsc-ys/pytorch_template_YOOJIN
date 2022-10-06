import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self):
        """ Configuration for the model """
        # Initializing weights for the sake of weights' randomness
        # https://stats.stackexchange.com/questions/45087/why-doesnt-backpropagation-work-when-you-initialize-the-weights-the-same-value
        super().__init__()
        
        # Convolutional Layer configuration
        self.conv1 = nn.Conv2d(
            in_channels = 1, 
            out_channels = 32, 
            kernel_size = 3, 
            stride = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3, 
            stride = 1)
        
        # Designating dropout configuration
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Fully Connected Layers configuration
        self.fc1 = nn.Linear(
            in_features = 9216, 
            out_features = 128
            )
        self.fc2 = nn.Linear(
            in_features = 128, 
            out_features = 10 # MNIST labels 0 ~ 9
            )

    def forward(self, x):
        """ Forward pass of the model using predefined configuration settings on ConvolutionalNeuralNetwork """
        # Stack Convolutional layers
        x = self.conv1(x) # using class' own function
        x = F.relu(x) # using torch.nn.functional function
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flattens input by reshaping it into a one-dimensional tensor
        x = torch.flatten(x, start_dim = 1) # using torch's function: https://pytorch.org/docs/stable/generated/torch.flatten.html
        
        # Go through Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
