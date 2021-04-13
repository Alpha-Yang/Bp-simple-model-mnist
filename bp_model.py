import torch
import torch.nn as nn
import os
from util import get_data

batch_size = 50
input_size = 784
hidden_size = 500
num_classes = 10
epochs = 50
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class BpNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(BpNet,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,num_classes)
        
    def forward(self,x):
        y=self.fc1(x)
        y=self.relu(y)
        y=self.fc2(y)
        return y

def train():
    trainDataset, testDataset = get_data()
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=batch_size, 
        shuffle=False,
    )
    model = BpNet(input_size, hidden_size, num_classes)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    # training our bp models

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(-1,28*28)
            # forward pass
            y_pred = model(x)
            # compute loss
            loss = lossfunc(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0: 
                print('Epoch [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, loss.item()))
    
    saved_model = 'bp_model.pt'
    torch.save(model, saved_model)

def test():
    trainDataset, testDataset = get_data()
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=batch_size, 
        shuffle=False,
    )
    saved_model = 'bp_model.pt'
    pre_model = torch.load(saved_model)
    lossfunc = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1,28*28)
            output = pre_model(data)
            test_loss += lossfunc(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    train()
    test()