import numpy as np 
import matplotlib.pyplot as plt 

c = [4,5,6,7,8,9,10]

qubit_train = [0.984, 0.979, 0.958, 0.959, 0.954, 0.946, 0.946]
qudit_train = [0.972, 0.968, 0.943, 0.935, 0.93, 0.912, 0.898]

qubit_test = [0.948, 0.945, 0.91, 0.879, 0.874, 0.82, 0.802]
qudit_test = [0.939, 0.939, 0.909, 0.878, 0.864, 0.813, 0.787]

plt.plot(c, qubit_train, label='dim=2')
plt.plot(c, qudit_train, label='dim=c')
plt.xlabel('number of classes')
plt.ylabel('training accuracy')
plt.legend()
plt.grid()
plt.show()