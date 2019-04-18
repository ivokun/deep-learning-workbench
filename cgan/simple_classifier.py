import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from dataloader import samplingloader
from dataloader import split_data

class Net(nn.Module):
  
  def __init__(self, input):
    super(Net, self).__init__()
    self.input = input

    self.net = nn.Sequential(
      nn.Linear(self.input, self.input),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(self.input,30),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(30, 1),
      nn.Sigmoid()
    )
    
  def forward(self, x):
    return self.net(x)


def train(net, train_loader, optimizer, epoch):
    """Create the training loop"""
    net.train()
    criterion = nn.BCELoss()
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader):
        features = data['features']
        target = data['target']

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_index % 6 == 5:  # print every 6 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_index + 1, running_loss / 6))
            running_loss = 0.0

def test(net, test_loader):
    """Test the DNN"""
    net.eval()
    criterion = nn.BCELoss()  # https://pytorch.org/docs/stable/nn.html#bceloss
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            features = data['features']
            target = data['target']
            output = net(features)
            # Binarize the output
            pred = output.apply_(lambda x: 0.0 if x < 0.5 else 1.0)
            test_loss += criterion(output, target)  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set:\n\tAverage loss: {:.4f}'.format(test_loss))
    print('\tAccuracy: {}/{} ({:.0f}%)\n'.format(
            correct,
            (len(test_loader) * test_loader.batch_size),
            100. * correct / (len(test_loader) * test_loader.batch_size)))

dataset = 'breast-cancer'
data = samplingloader(dataset, 'arff')
train_data, test_data, features_num = split_data(data)
torch.manual_seed(42)
net = Net(features_num).double()
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5, nesterov=False)
epochs = 10

print("Training Start")
for epoch in range(1, epochs+1):
  train(net, train_data, optimizer, epoch)
  test(net, test_data)

torch.save(net.state_dict(), 'models/'+dataset+'_without_sampling')

print("DONE")