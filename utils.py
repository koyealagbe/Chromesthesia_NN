import numpy as np
import time
import random

# Function: normalize
def normalize(x,mu,sigma):
  return (x-mu)/sigma

# Function: ReLU
def ReLU(x):
  return np.maximum(0,x)

# Function: dReLU
def dReLU(x):
  x[(x<=0)] = 0
  x[(x>0)] = 1
  return x

# Function: output
def output(z,y,lamb,w):
  yHat = 1/(1+np.exp(-z))
  if y == 1:
    loss = -np.log(yHat)
  else:
    loss = -np.log(1-yHat)
  reg = 0
  reg += np.sum(np.square(w))
  reg *= (lamb/2)
  loss += reg
  return yHat,loss

# Function: dOutput
def dOutput(y,yHat):
  return yHat - y

# Function: runNetwork
def runNetwork(X,Y,W,b,lamb):
  # Get dimensions
  (m,n) = X.shape
  k = Y.shape[1]
  l = len(W)-1 # number of hidden layers

  # Initialize values
  DW = [] # derivatives w.r.t weight matrices
  Db = [] # derivatives w.r.t bias vectors
  for i in range(len(W)):
    DW.append(np.zeros(W[i].shape))
    Db.append(np.zeros(b[i].shape))

  YHat = np.zeros(Y.shape) # predicted values
  J = 0 # total loss
  
  # Iterate through examples
  for i in range(m):
    # 1. Forward propagation
    x = X[i].reshape((n,1))
    y = Y[i].reshape((k,1))
    z = []
    a = []
    z.append(W[0]@x + b[0])
    for j in range(l):
      a.append(ReLU(z[j]))
      z.append(W[j+1]@a[j] + b[j+1])
    yHat,loss = output(z[l],y,lamb,W[l])
    YHat[i] = yHat.reshape(1,k)

    # 2. Add to total cost
    J += loss/m

    # 3. Back propagation
    d = np.array([])
    for j in range(l+1):
      if j == 0:
        d = dOutput(y,yHat) + np.sum(lamb*W[l])
      else:
        d = (W[l+1-j].T @ d) * dReLU(z[l-j])
      Db[l-j] += d/m
      if j != l:
        DW[l-j] += (d @ a[l-j-1].T)/m
      else:
        DW[l-j] += (d @ x.T)/m
    '''
    d3 = dOutput(y,yHat)
    Db[2] += d3/m
    DW[2] += (d3 @ a2.T)/m

    d2 = (W[2].T @ d3) * dReLU(z2)
    Db[1] += d2/m
    DW[1] += (d2 @ a1.T)/m

    d1 = (W[1].T @ d2) * dReLU(z1)
    Db[0] += d1/m
    DW[0] += (d1 @ x.T)/m
    '''

  return YHat,J,DW,Db

# Function: predict
def predict(X,Y,W,b,lamb):
  l = len(W)-1
  (m,n) = X.shape
  k = Y.shape[1]
  YHat = np.zeros(Y.shape)
  J = 0 
  for i in range(m):
    x = X[i].reshape((n,1))
    y = Y[i].reshape((k,1))
    z = []
    a = []
    z.append(W[0]@x + b[0])
    for j in range(l):
      a.append(ReLU(z[j]))
      z.append(W[j+1]@a[j] + b[j+1])
    yHat,loss = output(z[l],y,lamb,W[l])
    YHat[i] = yHat.reshape(1,k)
    J += loss/m
  return YHat,J

# Function: gradDesc
def gradDesc(X,Y,X_v,Y_v,W,b,lamb,alpha,nIterations):
  hist = np.zeros(nIterations)
  hist_v = np.zeros(nIterations)
  for i in range(nIterations):
    _,J,DW,Db = runNetwork(X,Y,W,b,lamb)
    hist[i] = J
    _,hist_v[i] = predict(X_v,Y_v,W,b,lamb)
    for l in range(len(W)):
      W[l] = W[l]*(1-(alpha*lamb/X.shape[0])) - (alpha*DW[l])
      b[l] = b[l] - (alpha*Db[l])
    #if i == 100000:
      #alpha = 0.001
  return hist,hist_v,W,b

# Function: accuracy
def accuracy(yHat,y):
  correct = np.sum(y[yHat>=0.5]) + np.sum((1-y)[yHat<0.5])
  return correct/y.shape[0]

# Function: F1
def F1(yHat,y):
  tp = np.sum(((yHat>=0.5) & (y==1)))
  fp = np.sum(((yHat>=0.5) & (y==0)))
  fn = np.sum(((yHat<0.5) & (y==1)))
  return tp / (tp + (1/2)*(fp+fn))

# Function: errorSummary
def errorSummary(X,Y,W,b,lamb):
  YHat,J = predict(X,Y,W,b,lamb)
  print("Cross-entropy loss:",J)
  print("Accuracy:",accuracy(YHat,Y))
  print("F1 Score:",F1(YHat,Y))



