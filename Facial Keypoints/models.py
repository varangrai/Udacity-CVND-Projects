import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# we have input of 1 x 224 x 224 image, ie converteed to gray scale as well as rescaled
# the output corresponding to this is a 68 x 2 matrix hence last layer should have this shape

class Net(nn.Module):
     
        
        #the channels grow in power of 2
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # 224 - 3 + 1
        self.pool = nn.MaxPool2d(2, 2)    #(222-2)/2 +1
        self.conv2 = nn.Conv2d(8, 64, 3)    # 111 -2
        self.pool2 = nn.MaxPool2d(2,2)       # (109 - 2)/2 +1
        self.conv3 = nn.Conv2d(64,128,5)     # 54 -5 + 1 =50
        self.pool3 = nn.MaxPool2d(2,2)     # 50 -2)/2 +1 = 25
        self.conv4 = nn.Conv2d(128,256,5)     # 25 -5 + 1 = 21
        self.pool4 = nn.MaxPool2d(2,2)     # (21 -2)/2 + 1 = 10
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(10*10*256, 5000)
        # dropout with p=0.4
        self.cl_drop = nn.Dropout(p=0.4)
        self.fc1_drop = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(5000,500)
        self.fc2_drop = nn.Dropout(p=0.5)
        # finally, create 10 output channels (for the 10 classes)
        self.fc3 = nn.Linear(500, 136)

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.cl_drop(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.cl_drop(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.cl_drop(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.cl_drop(x)
        # prep for linear layer
    
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # final output
        return x