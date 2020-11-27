import numpy as np
from matplotlib import pyplot as plt
import os
import utils

### LOAD AND PROCESS DATA ###
data_train = np.loadtxt(os.path.join('Data/pink','pink_train.csv'),delimiter=',')
data_val = np.loadtxt(os.path.join('Data/pink','pink_val.csv'),delimiter=',')
X_train = data_train[:,:3]
Y_train = data_train[:,3]
X_val = data_val[:,:3]
Y_val = data_val[:,3]
#X_test = data_test[:,:3]
#Y_test = data_test[:,3]

(m,n) = X_train.shape # m is number of examples, n is number of features
l = 3 # number of hidden layers
s = 10 # size of the hidden layers
k = 1 # number of classes
Y_train = Y_train.reshape(m,k)
Y_val = Y_val.reshape(X_val.shape[0],k)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
print(mu)
print(sigma)
X_train = utils.normalize(X_train,mu,sigma)
X_val = utils.normalize(X_val,mu,sigma)

### INITIALIZE PARAMETERS ###
np.random.seed(1)
W = []
b = []

W.append(np.random.rand(s,n) * np.sqrt(2/n))
b.append(np.zeros((s,1)))
for i in range(l-1):
  W.append(np.random.rand(s,s) * np.sqrt(2/s))
  b.append(np.zeros((s, 1)))
W.append(np.random.rand(k,s) * np.sqrt(2/s))
b.append(np.zeros((k, 1)))

### RUN GRADIENT DESCENT ###
lamb = 0
alpha = 0.005
nIterations = 100000
J,J_val,W,b = utils.gradDesc(X_train,Y_train,X_val,Y_val,W,b,lamb,alpha,nIterations)

for i in range(len(W)):
  print("W"+str(i+1)+":")
  print(W[i])
  print("b"+str(i+1)+":")
  print(b[i])
print()


print("Training Error Summary:")
utils.errorSummary(X_train,Y_train,W,b,lamb)
print()
print("Validation Error Summary:")
utils.errorSummary(X_val,Y_val,W,b,lamb)

xs = np.arange(1,nIterations+1)
plt.plot(xs,J,color='b')
plt.plot(xs,J_val,color='r')
plt.show()