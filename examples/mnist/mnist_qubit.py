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
manualSeed = 3
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True

N_train = 1000 #number of samples from the training set
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
            qf.Linear(784, 10)
        )

        self.init = qf.H(dim=dim, index=range(wires))
        self.qencoder = qf.RZ(dim=dim, index=range(wires))

        self.U = qf.U(dim=dim, wires=wires)

    def forward(self, x):
        x = x.reshape((1, 784))
        x = self.encoder(x)
        x = x.flatten()

        y = State('0-0-0-0-0-0-0-0-0-0', device=x.device)
        y = self.init(y)
        y = self.qencoder(y, param=x)
        y = self.U(y)

        return y

#Instatiate the circuit and define the optimizer
device = 'cuda'
dim = 2
wires = 10
model = Circuit(dim=dim, wires=wires).to(device)
optimizer = qf.optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999))

#Define the target states
targets = []
for i in range(10):
    state = ''
    for j in range(10):
        if j == i:
            state += '1'
        else:
            state += '0'
        if j < 9:
            state += '-'
    targets.append(State(state, dim=2, device=device))

targets_arg = []
for i in range(10):
    targets_arg.append(torch.argmax(abs(targets[i])))

train_list = []
test_list = []
#Let's train the model
for epoch in range(8):
    acc_train = 0
    acc_test = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)

        output = model(inputs)
        target = targets[labels[0]].reshape(output.shape)
        F = qf.fidelity(target, output)
        loss = (1-F)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, m = qf.measure(output, index=range(10), dim=dim, wires=wires)
        predict = torch.argmax(m).item()
        if predict == targets_arg[labels[0]]:
            acc_train += 1

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            output = model(inputs)
            _, m = qf.measure(output, index=range(10), dim=dim, wires=wires)

            predict = torch.argmax(m)
            if predict == targets_arg[labels[0]]:
                acc_test += 1

    acc_train = acc_train/N_train
    acc_test = acc_test/N_test
    print(epoch, acc_train, acc_test)
    train_list.append(acc_train)
    test_list.append(acc_test)

np.save('qubit_train_%i.npy' % manualSeed, train_list)
np.save('qubit_test_%i.npy' % manualSeed, test_list)