import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
from dataset import MyDataset
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy


#get the number of unique colors and piece id son that we know how many outputs we need
df = pd.read_csv("lego_pieces.csv")
NUM_PIECES = df['pieceid'].nunique()
NUM_COLORS = df["colorid"].nunique()


labels = 'lego_pieces.csv'
dataset = MyDataset(labels_csv=labels)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining for testing

training_data, test_data = random_split(dataset, [train_size, test_size])

device = ('cuda' if torch.cuda.is_available() else 'cpu') # choose the device depending on GPU compatibility

# Define the structure of the CNN model.
class MYCNN(nn.Module):
    def __init__(self, num_pieceid, num_colorid):
        super(MYCNN, self).__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  #3 in 32 out padding 1 is used to keep the size 64x64
            nn.ReLU(),
            nn.MaxPool2d(2), #halves size (32x32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  #32 in 64 out size still 32x32 thanks to padding
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves size(16x16)
)
        self.flatten = nn.Flatten()

        #the fully connected layer is common to all 3 heads to optimize computation
        self.fc_common = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU()
        )

        #we will need 3 output heads (one for each label)
        self.fc_pieceid = nn.Linear(256, num_pieceid)    #need a neuron for each piece/color
        self.fc_colorid = nn.Linear(256, num_colorid)    
        self.fc_plant = nn.Linear(256, 1)                #need a single neuron (activated/not)

    def forward(self, x):
        #forward as with pannone but instead of classifying after the fc each head will classify based on it, a dict with the predicted values is returned
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_common(x)

        pieceid_out = self.fc_pieceid(x)
        colorid_out = self.fc_colorid(x)
        plant_out = self.fc_plant(x)

        return {
            "pieceid": pieceid_out,
            "colorid": colorid_out,
            "plant": plant_out.squeeze(1)
        }
# At this point, create an instance of the model.
model = MYCNN(num_pieceid=NUM_PIECES, num_colorid=NUM_COLORS)
model = model.to(device)

epochs = 2 # check the whole dataset twice
batch_size = 64 # check 64 data samples at once
learning_rate = 0.0001 # use a smaller learning rate in order to obtain more accurate results during the backpropagation phase

# Define the loss function for the classification model.
loss_fn_piece = nn.CrossEntropyLoss()  # for pieceid classification
loss_fn_color = nn.CrossEntropyLoss()  # for colorid classification
loss_fn_plant = nn.BCEWithLogitsLoss()  #for binary classification, more stable then just bce

# Use a dataloader for processing the data in batches.
train_dataloader = DataLoader(training_data, batch_size=batch_size) # dataloader for the training dataset
test_dataloader = DataLoader(test_data, batch_size=batch_size) # dataloader for the testing dataset

# Create the accuracy metric for estimating the goodness of the model.
#since we have 3 distinct labels we will need to compute the metric for ech label distinctly
metric_pieceid = MulticlassAccuracy(num_classes=NUM_PIECES).to(device) #to(device) is added because model is on cuda if not added an error will occur
metric_colorid = MulticlassAccuracy(num_classes=NUM_COLORS).to(device)
metric_plant = BinaryAccuracy().to(device) 

# Define the optimizer for gradient descent.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # perform stochastic gradient descent on the parameters

# Define the training loop for the training stage.
def training_loop(dataloader, model, optimizer):
    dataset_size = len(dataloader)
    for batch, (X, labels) in enumerate(dataloader):
        X = X.to(device) #moves from cpu to gpu otherwise an error will occut since the model is cuda
        labels = {k: v.to(device) for k, v in labels.items()} #same as above
        outputs = model(X)

        #we have 3 different heads so 3 loss fn so we have to compute our loss as the sum of the single losses
        loss_piece = loss_fn_piece(outputs['pieceid'], labels['pieceid'])
        loss_color = loss_fn_color(outputs['colorid'], labels['colorid'])
        loss_plant = loss_fn_plant(outputs['plant'].squeeze(), labels['plant'])

        loss = loss_piece + loss_color + loss_plant
        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)

            #compute each acc
            acc_piece = metric_pieceid(outputs['pieceid'], labels['pieceid'])
            acc_color = metric_colorid(outputs["colorid"], labels["colorid"])
            acc_plant = metric_plant(outputs["plant"], labels["plant"])

            print(f"Batch {batch}, Loss: {loss:.4f}, Progress: [{current} / {dataset_size}]")
            print(f"Accuracy - PieceID: {acc_piece:.2f}%, ColorID: {acc_color:.2f}%, Plant: {acc_plant:.2f}%")

    #final training acc
    acc_piece = metric_pieceid.compute()
    acc_color = metric_colorid.compute()
    acc_plant = metric_plant.compute()
    print(f"Accuracy of the current batch: piece:{acc_piece}, color: {acc_color}, plant: {acc_plant}")
    # Reset metriche per futuri cicli
    metric_pieceid.reset()
    metric_colorid.reset()
    metric_plant.reset()
# Define the testing loop for the testing stage.
def testing_loop(dataloader, model):
    with torch.no_grad():
        for X, labels in dataloader:
            X = X.to(device) #same as above
            labels = {k: v.to(device) for k, v in labels.items()} #same as above
            outputs = model(X)

            # Predictions for each head
            preds_piece = outputs['pieceid'].argmax(dim=1)
            preds_color = outputs['colorid'].argmax(dim=1)
            preds_plant = (outputs['plant'] > 0.5).long()

            # Update metrics
            metric_pieceid.update(preds_piece, labels['pieceid'])
            metric_colorid.update(preds_color, labels['colorid'])
            metric_plant.update(preds_plant, labels['plant'].long())

    # Compute and print final accuracies
    acc_piece = metric_pieceid.compute()
    acc_color = metric_colorid.compute()
    acc_plant = metric_plant.compute()

    print(f"Final testing accuracy: piece: {acc_piece}, color: {acc_color}, plant: {acc_plant}")

    # Reset metrics for next evaluation
    metric_pieceid.reset()
    metric_colorid.reset()
    metric_plant.reset()

# Perform the actual model training and testing.
for e in range(epochs):
    print(f"\nEpoch {e+1}/{epochs}")
    training_loop(train_dataloader, model, optimizer)
    testing_loop(test_dataloader, model)
print("Done")