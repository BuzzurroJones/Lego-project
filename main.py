import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics

device = ('cuda' if torch.cuda.is_available() else 'cpu') # choose the device depending on GPU compatibility

# Start by importing the training and testing datasets.
training_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor()) # (down)load, and convert to tensor, the training data of MNIST and into a 'data' folder
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor()) # (down)load, and convert to tensor, the testing data of MNIST and into a 'data' folder

# Define the structure of the CNN model.
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sequential CNN for the convolution and pooling layers.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3), # use 2D convolution on grayscale images (1 input channel, arbitrary output channels) using a 3x3 kernel
            nn.ReLU(), # set an activation function
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.ReLU()
        )
        self.flatten = nn.Flatten() # to flatten the feature map for the fully connected layer
        # self.flatten = torch.flatten(x, 1) # to flatten the feature map for the fully connected layer within the forward function
        # Create a MLP for the fully connected layer.
        self.mlp = nn.Sequential(
            nn.Linear(5760, 10), # fix by raising an error on a random tensor (LazyLinear can be used but it is not fully functional yet)
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    def forward(self, x):
        x = self.cnn(x) # create a 2D feature part using the convolutional neural network
        x = self.flatten(x) # flatten the feature map so that it can be used by the multilayer perceptron
        logits = self.mlp(x) # use the MLP for classification by exploiting the features extracted from the CNN
        return logits

# At this point, create an instance of the model.
model = MyCNN()
# fake_input = torch.rand((1, 1, 28, 28)) # create a random tensor of type (b, c, w, h) in order to print the size of the flattened feature map for fixing the CNN model
# model(fake_input)

# Start by defining the hyperparameters of the model.
epochs = 2 # check the whole dataset twice
batch_size = 64 # check 64 data samples at once
learning_rate = 0.0001 # use a smaller learning rate in order to obtain more accurate results during the backpropagation phase

# Define the loss function for the classification model.
loss_function = nn.CrossEntropyLoss() # choose cross entropy loss

# Use a dataloader for processing the data in batches.
train_dataloader = DataLoader(training_data, batch_size=batch_size) # dataloader for the training dataset
test_dataloader = DataLoader(test_data, batch_size=batch_size) # dataloader for the testing dataset

# Create the accuracy metric for estimating the goodness of the model.
metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

# Define the optimizer for gradient descent.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # perform stochastic gradient descent on the parameters

# Define the training loop for the training stage.
def training_loop(dataloader, model, loss_function, optimizer):
    dataset_size = len(dataloader)
    # Start by loading the data batch from the disk.
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X) # get the predictions of the model
        # Compute the prediction error using the loss function.
        loss = loss_function(pred, y)
        # Perform backpropagation in order to minimize future losses.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 500 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss:{loss}, [{current} /{dataset_size}]")
            acc = metric(pred, y)
            print(f"Accuracy of the current batch: {acc}")
    # Print the final training accuracy.
    acc = metric.compute()
    print(f"Final training accuracy: {acc}")
    metric.reset() # reset for future loops

# Define the testing loop for the testing stage.
def testing_loop(dataloader, loss_function, model):
    # Remember to disable the weights update.
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            acc = metric(pred, y)
    acc = metric.compute()
    print(f"Final testing accuracy: {acc}")
    metric.reset()

# Perform the actual model training and testing.
for e in range(epochs):
    print(f"Epoch: {e}")
    training_loop(train_dataloader, model, loss_function, optimizer)
    testing_loop(test_dataloader, loss_function, model)
print("Done")