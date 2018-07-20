import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.datasets as data
import torchvision.transforms as transforms

from models import MultiLayerNeuralNetworks

# hyper params
batch_size = 128
epochs = 10
input_size = 784
num_classes = 10
learning_rate = 0.001

# load MNIST dataset
train_dataset = data.MNIST(root='./mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_dataset = data.MNIST(root='./mnist',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)
# dataset loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# create models
model = MultiLayerNeuralNetworks(input_feature=784, hidden_size=100, num_classes=10)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch: [%d/%d], Batches: [%d/%d], Loss: %.4f" %
                  (epoch+1, epochs, i+1, len(train_dataset) // batch_size, loss.item()))

# Test the model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += torch.sum(labels == predicted)

print("Accuracy of the model on the test data is: %d %%" % (100 * correct / total))
