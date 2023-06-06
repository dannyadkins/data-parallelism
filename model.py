import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, model_size=1, dropout=None):
        super(CNN, self).__init__()
        # batch is a 4d tensor of shape (batch_size, 1, 28, 28)
        self.model_size = model_size
        # predict the digit likelihood 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=model_size, kernel_size=3, stride=1, padding=1)
        # batch is a 4d tensor of shape (batch_size, 16, 28, 28)
        self.conv2 = nn.Conv2d(in_channels=model_size, out_channels=model_size*2, kernel_size=3, stride=1, padding=1)
        # batch is a 4d tensor of shape (batch_size, 32, 28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # batch is a 4d tensor of shape (batch_size, 32, 14, 14)
        self.fc1 = nn.Linear(in_features=model_size*2*14*14, out_features=10)
        # batch is a 2d tensor of shape (batch_size, 10)
        self.dropout = dropout
        if (dropout is not None):
            self.dropout1 = nn.Dropout(dropout)

        # last steps to get the digit likelihood
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, batch):
        batch = self.conv1(batch)
        batch = self.conv2(batch)
        batch = self.pool(batch)
        batch = batch.view(-1, self.model_size*2*14*14)
        batch = self.fc1(batch)
        if (self.dropout is not None):
            batch = self.dropout1(batch)
        batch = self.softmax(batch)
        return batch 