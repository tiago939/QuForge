import numpy as np 
import matplotlib.pyplot as plt

train2 = []
test2 = []
train10 = []
test10 = []
for i in range(1,2):
    qudit_train = np.load('qudit_train_%i.npy' % i)
    qubit_train = np.load('qubit_train_%i.npy' % i)
    qudit_test= np.load('qudit_test_%i.npy' % i)
    qubit_test = np.load('qubit_test_%i.npy' % i)

    train2.append(qubit_train)
    test2.append(qubit_test)
    train10.append(qudit_train)
    test10.append(qudit_test)

qubit_mean = np.mean(train2, axis=0)
qubit_min = np.min(train2, axis=0)
qubit_max = np.max(train2, axis=0) 
qudit_mean = np.mean(train10, axis=0)
qudit_min = np.min(train10, axis=0)
qudit_max = np.max(train10, axis=0)

epochs = range(9)
#plt.fill_between(epochs, qubit_min, qubit_max, alpha=0.25, color='red')
#plt.fill_between(epochs, qudit_min, qudit_max, alpha=0.25, color='blue')
plt.plot(qudit_mean, label='dim=10')
plt.plot(qubit_mean, label='dim=2')
plt.grid()
plt.ylabel('Training accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

qubit_mean = np.mean(test2, axis=0)
qubit_min = np.min(test2, axis=0)
qubit_max = np.max(test2, axis=0) 
qudit_mean = np.mean(test10, axis=0)
qudit_min = np.min(test10, axis=0)
qudit_max = np.max(test10, axis=0)

epochs = range(9)
#plt.fill_between(epochs, qubit_min, qubit_max, alpha=0.25, color='red')
#plt.fill_between(epochs, qudit_min, qudit_max, alpha=0.25, color='blue')
plt.plot(qudit_mean, label='dim=10')
plt.plot(qubit_mean, label='dim=2')
plt.grid()
plt.ylabel('Test accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()