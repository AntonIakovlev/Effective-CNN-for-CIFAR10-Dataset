# import libraries
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import time
#%matplotlib inline

#---

# prepare input data

valid_size = 0.1
batch_size = 128
numWorkers = 16
numEpoch = 40
logStep = 200

iterations = []
losses = []
accuracies = []
iterations_val = []
losses_val = []
accuracies_val = []

pathToDatasets = '/datasets/'
pathToData = pathToDatasets
datasetName = 'CIFAR10'
pathToData += datasetName + '/'

transformationsAugmented = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])

transformations = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

train_dataset = torchvision.datasets.CIFAR10(root=pathToData, train=True, download=True, transform=transformationsAugmented)
#val_dataset = torchvision.datasets.CIFAR10(root=pathToData, train=True, download=True, transform=transformations)
test_dataset = torchvision.datasets.CIFAR10(root=pathToData, train=False, download=True, transform=transformations)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=numWorkers)
valloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=numWorkers)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=numWorkers)

#---

class CNN2(torch.nn.Module):
    def __init__(self, numClasses=10):
        super(CNN2, self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=False),
                    nn.BatchNorm2d(64)
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
                    nn.BatchNorm2d(64)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, bias=False),
                    nn.BatchNorm2d(32)
                    )
        self.layer4 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False),
                    nn.BatchNorm2d(32)
                    )
        
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(32 * 10 * 10, 10)
        self.criterion = nn.CrossEntropyLoss() # loss 

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.pool(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.fc(x.view(-1, 32 * 10 * 10))
        return x

#---

def defaultInit(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    print("Model initalised successfully")


def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            # count total images in batch
            total += labels.size(0)
            # count number of correct images
            correct += (predicted == labels).sum()

    test_acc = correct.item()/float(total)

    print("Accuracy on Test Set: %.4f" % test_acc)
    

def evalValidation(model, device, valLoader, n_iter):

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = []

        for data in valLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            total_loss.append(model.criterion(outputs, labels).item())

            _, predicted = torch.max(outputs.data, 1)
            # count total images in batch
            total += labels.size(0)
            # count number of correct images
            correct += (predicted == labels).sum()

    acc = correct.item()/float(total)

    avg_loss = sum(total_loss)/float(len(total_loss))
    
    print("Iteration: %d. Accuracy on Validation Set: %.4f, Average Loss on Validation Set: %.4f" % (n_iter, acc, avg_loss))
    iterations_val.append(n_iter)
    losses_val.append(avg_loss)
    accuracies_val.append(acc)
    
    return avg_loss, acc


def getAccuracy(outputs, labels, num):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return correct.item()/float(num)

#---
use_cuda = False

device = torch.device("cuda" if use_cuda else "cpu")

model = CNN2()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {}".format(pytorch_total_params))

defaultInit(model)
model = model.to(device)

optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[15,29,32], gamma=0.5)

n_iter = 0

for epoch in range(1, numEpoch + 1):
    # do validation here
    val_loss, val_acc = evalValidation(model, device, valloader, n_iter)
        
    t00 = time.time()
    scheduler.step()
    
    model.train()
    
    for i, data in enumerate(trainloader):
    
        t0 = time.time()
        
        optimiser.zero_grad()
        
        inputs, labels = data[0].to(device), data[1].to(device)
                
        # forward pass
        outputs = model(inputs)
                
        loss = model.criterion(outputs, labels)
        
        # backward pass
        loss.backward()

        # evaluate trainable parameters
        optimiser.step()
        
        acc = getAccuracy(outputs, labels, inputs.shape[0])
        
        n_iter += inputs.shape[0]
        
        if i % logStep == 0:
            print("Epoch: {}, Step: {}, Accuracy: {:.4f}, Loss: {:.4f}  -- Time: {} s".format(epoch,i,acc, time.time() - t0, loss.item()))
            iterations.append(n_iter)
            losses.append(loss.item())
            accuracies.append(acc)
        
    print("Epoch took: %.3f s " % (time.time() - t00))

# plot learning information
plt.figure(1)
plt.title('Loss function for the training set')
plt.scatter(iterations, losses, c='r')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.figure(2)
plt.title('Accuracy function for the training set')
plt.scatter(iterations, accuracies, c='b')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure(3)
plt.title('Loss function for the valuation set')
plt.scatter(iterations_val, losses_val)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.figure(4)
plt.title('Accuracy function for the valuation set')
plt.scatter(iterations_val, accuracies_val)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
         
plt.show()

# do test here
test(model, device, testloader)

#---