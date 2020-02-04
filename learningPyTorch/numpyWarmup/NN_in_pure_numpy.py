# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 
import pdb

#config
inputSize = 5
outputSize = 3
batchSize = 10
testSize = 5
epochs = 400
learningRate = 0.01
seed = 123456


#create functions
#Operator example in forward and backward (Mult)

def forwardMult(A,B):
    return np.matmul(A,B) 

def backwardMult(dC,A,B,dA,dB):
    dA += np.matmul(dC,np.matrix.transpose(B))
    dB += np.matmul(np.matrix.transpose(A),dC)
    
#Loss example in forward and backward (RMSE)
def forwardloss(predictedOutput,realOutput):
    if(predictedOutput.shape == realOutput.shape):
        loss = np.mean( 0.5*np.square(predictedOutput - realOutput))       
    else :
        print("Shape of arrays not the same")
    return loss

def backwardloss(predictedOutput,realOutput):
    if(predictedOutput.shape == realOutput.shape):
        deltaOutput = (predictedOutput - realOutput)/predictedOutput.size
    else :
        print("Shape of arrays not the same")
    return deltaOutput

#Optimizer example (SGD)
def updateweights(W,dW,learningRate):
    W -= learningRate * dW


#Generation of fake dataset - we generate random inputs and weights and calculate outputs
np.random.seed(seed)
inputArray = np.random.uniform(-5,5,(batchSize,inputSize))
weights = np.random.uniform(-5,5,(inputSize,outputSize))
outputArray = np.matmul(inputArray,weights)
inputTest = np.random.uniform(-5,5,(testSize,inputSize))
outputTest = np.matmul(inputTest,weights)


# inputArray
# outputArray
# inputTest
# outputTest


#initialization of NN by other random weights
nnWeights = np.random.uniform(-3,3,(inputSize,outputSize))
deltaweights = np.zeros((inputSize,outputSize))
deltainput = np.zeros((batchSize,inputSize))
deltaoutput = np.zeros((inputSize,outputSize))

weights; inputTest; nnWeights; outputArray
pdb.set_trace()

# In[10]:


#Comparing the dataset weights 
weights


# In[11]:


#with the NN weights
nnWeights


# In[12]:


#----------------------------------------------------------------------------------------------------
historyTrain=[] #Used to record the history of loss
historyTest=[]
i = 1

while i <= epochs:
    nnOutput = forwardMult(inputArray,nnWeights)
    lossTrain = forwardloss(nnOutput,outputArray)
    historyTrain.append(lossTrain)
    nnTest = forwardMult(inputTest,nnWeights)
    lossTest = forwardloss(nnTest,outputTest)
    historyTest.append(lossTest)
    if(i%100==0 & i != 0):
        print("Epoch: " + str(i) + " Loss (train): " + "{0:.3f}".format(lossTrain) + " Loss (test): " + "{0:.3f}".format(lossTest))
    deltaoutput = backwardloss(nnOutput,outputArray)
    backwardMult(deltaoutput,inputArray,nnWeights,deltainput,deltaweights)
    updateweights(nnWeights,deltaweights, learningRate)
    deltainput = np.zeros((batchSize,inputSize))
    deltaweights = np.zeros((inputSize,outputSize))
    deltaoutput = np.zeros((inputSize,outputSize))
    i = i+1
        
#----------------------------------------------------------------------------------------------------
# plt.plot(historyTrain)
# plt.plot(historyTest)
# plt.title('RMSE loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train','test'], loc='upper right')
# plt.show()

# In[14]:


# Original weights used to generate the dataset
# weights


# In[15]:


# Learned weights of the NN
#nnWeights

