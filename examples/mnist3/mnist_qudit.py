# # First, we import the libaries
import sys
sys.path.append('../')
import quforge.quforge as qf
from quforge.quforge import State as State

#Load the dataset
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

#random seed
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True

N_train = 100 #number of samples from the training set
N_test = 100 #number of samples from the test set

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='../../datasets/', train=True, download=True, transform=transform_train)
indices = torch.arange(N_train)
trainset = torch.utils.data.dataset.Subset(trainset, indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='../../datasets/', train=False, download=True, transform=transform_train)
indices = torch.arange(N_test)
testset = torch.utils.data.dataset.Subset(testset, indices)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

#Now, we define the hybrid model

class Circuit(qf.Module):
    def __init__(self, dim, wires):
        super(Circuit, self).__init__()

        self.encoder = qf.Sequential(
            qf.Linear(784, dim**wires)
        )

        self.init = qf.H(dim=dim, index=range(wires))

        self.U = qf.U(dim=dim, wires=wires)

    def forward(self, x):
        x = x.reshape((1, 784))
        zeros = torch.zeros(1, 1000 - 784, device=x.device)
        x = torch.cat((x, zeros), dim=1).reshape((1000, 1)).type(torch.complex64)
        x = x/(torch.sum(abs(x)**2)**0.5)
        x = self.U(x)

        return x

#Instatiate the circuit and define the optimizer
device = 'cuda'
dim = 10
wires = 3
model = Circuit(dim=dim, wires=wires).to(device)
optimizer = qf.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999))

#Define the target states
targets = []
for i in range(10):
    state = str(i)
    targets.append(State(state, dim=10, device=device))

train_list = []
test_list = []
#Let's train the model
for epoch in range(8):
    acc_train = 0
    acc_test = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)

        output = model(inputs)
        _, m = qf.measure(output, index=[0], dim=dim, wires=wires)
        target = targets[labels[0]]
        loss = torch.mean(abs(target.flatten()-m.flatten()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, m = qf.measure(output, index=[0], dim=dim, wires=wires)
        predict = torch.argmax(m).item()
        if predict == labels[0]:
            acc_train += 1

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            output = model(inputs)
            _, m = qf.measure(output, index=[0], dim=dim, wires=wires)

            predict = torch.argmax(m)
            if predict == labels[0]:
                acc_test += 1

    acc_train = acc_train/N_train
    acc_test = acc_test/N_test
    print(epoch, acc_train, acc_test)
    train_list.append(acc_train)
    test_list.append(acc_test)

np.save('qudit_train_%i.npy' % manualSeed, train_list)
np.save('qudit_test_%i.npy' % manualSeed, test_list)

torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }, 'qudit_%i.pt' % manualSeed)