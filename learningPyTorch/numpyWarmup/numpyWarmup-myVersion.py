# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pdb

#------------------------------------------
def showTwoMatricesAsBitMaps(m1, m2):

   fig, (p1, p2) = plt.subplots(1,2)
   img1 = p1.imshow(m1)
   img2 = p2.imshow(m2)
   fig.show()
    
#------------------------------------------
# --- fromWeb.py
# N, D_in, H, D_out = 64, 1000, 100, 10
# learning_rate = 1e-6
# for t in range(500):


batchSize = 44           # number of samples (observations) to calculate over.  

inputDimension = 1000
hiddenDimension = 100
outputDimension = 10
reps = 500
learningRate = 1e-6

inputDimension = 10
hiddenDimension = 100
outputDimension = 10
reps = 500
learningRate = 1e-6


samples <- 200

y = np.random.randn(


# Create random input and output data
x = np.random.randn(batchSize, inputDimension)

#x = x - np.min(x)
#x = x/np.max(x)
# x[:, stripStart:stripEnd] = 1

y = np.random.randn(batchSize, outputDimension)
stripStart = round(batchSize/2)
stripEnd = stripStart + 10
y[stripStart:stripEnd, :] = np.max(y)
y = y - np.min(y)
y = y/np.max(y)

# Randomly initialize weights
w1 = np.random.randn(inputDimension, hiddenDimension)
w2 = np.random.randn(hiddenDimension, outputDimension)

# relu: rectified linear unit, an activation function, defined as y = max(0, x)
# the activation function of a node defines the output of that node given
# an input or set of inputs.

for t in range(reps):
    h = x.dot(w1)     # Forward pass: compute predicted y
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()      # Compute and print loss
    #if t % 5000 == 0 & t != 0:
    #    pdb.set_trace()

    if t % 100 == 0:
        print(t, loss)
        showTwoMatricesAsBitMaps(y, y_pred)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learningRate * grad_w1
    w2 -= learningRate * grad_w2
