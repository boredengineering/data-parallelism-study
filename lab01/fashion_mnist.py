import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Parse input arguments
parser = argparse.ArgumentParser(description='Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs that meet target before stopping')

args = parser.parse_args()

# Standard convolution block followed by batch normalization 
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1),
                               padding='same', bias=False), 
                               nn.BatchNorm2d(output_channels), 
                               nn.ReLU()
        )
    def forward(self, x):
        out = self.cbr(x)
        return out

# Basic residual block
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1,1),
                               padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
        
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        out = out + residual
        
        return out
    
# Overall network
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [1, 16, 160, 320, 640]

        self.input_block = cbrblock(nChannels[0], nChannels[1])
        
        # Module with alternating components employing input scaling
        self.block1 = conv_block(nChannels[1], nChannels[2], 1)
        self.block2 = conv_block(nChannels[2], nChannels[2], 0)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], 1)
        self.block4 = conv_block(nChannels[3], nChannels[3], 0)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], 1)
        self.block6 = conv_block(nChannels[4], nChannels[4], 0)
        
        # Global average pooling
        self.pool = nn.AvgPool2d(7)

        # Feature flattening followed by linear layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)
        
        return out

def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        labels = labels.to(device)
        images = images.to(device)
        
        # Forward pass 
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Setting all parameter gradients to zero to avoid gradient accumulation
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Updating model parameters
        optimizer.step()

def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # Forward pass 
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Extracting predicted label, and computing validation loss and validation accuracy
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss
    
    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)
    
    return v_accuracy, v_loss

if __name__ == '__main__':
    
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  

    # Train only on 1/6 of the dataset
    train_subset = torch.utils.data.Subset(train_set, list(range(0, 10000)))
    test_subset = torch.utils.data.Subset(test_set, list(range(0, 10000)))
    
    # Training data loader
    train_loader = torch.utils.data.DataLoader(train_subset, 
                                               batch_size=args.batch_size, drop_last=True)
    # Validation data loader
    test_loader = torch.utils.data.DataLoader(test_subset,
                                              batch_size=args.batch_size, drop_last=True)

    # Create the model and move to GPU device if available
    num_classes = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WideResNet(num_classes).to(device)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)

    val_accuracy = []
    
    # TODO Step 3: initialize the variable total_time with a value of 0.
    
    for epoch in range(args.epochs):
        
        t0 = time.time()
        
        train(model, optimizer, train_loader, loss_fn, device)

        epoch_time = time.time() - t0
        # TODO Step 3: Compute total training time. Outside of the epoch loop, initialize the 
        # variable total_time with a value of 0. After each training epoch, increase the value 
        # of total_time by epoch_time. Append the cumulative training time after each epoch
        # to the print statement for metric summary below.
        
        # TODO Step 1: Compute and print image throughput (images per second) for each epoch.
        # Use the variable name images_per_sec to store the value. Note that within our train 
        # and test loader, we have set the parameter drop_last as True. This ensures that the 
        # total number of samples in our dataset is divisible by the batch size. 
        # len(train_loader) provides the number of batches per epoch. args.batch_size is the 
        # argument for the number of samples in a batch. epoch_time is the time taken to complete
        # one training epoch. 
        # Include the image throughput in the print statement below. 
        
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        
        val_accuracy.append(v_accuracy)
            
        print("Epoch = {:2d}: Epoch Time = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}".format(epoch+1, epoch_time, v_loss, val_accuracy[-1]))
        
        # TODO Step 2: implement early stopping. Use the validation accuracy array (val_accuracy) 
        # to assess if the number of epochs determined by "patience" exceeds a certain "target" accuracy.
        # When the stopping criteria is met, use the following print statement
        # print('Early stopping after epoch {}'.format(epoch + 1)), and break the epoch loop. 