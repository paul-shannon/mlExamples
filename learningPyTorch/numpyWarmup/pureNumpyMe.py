# coding: utf-8
#------------------------------------------------------------------------------------------------------------------------
#  from https://medium.com/datathings/a-neural-network-fully-coded-in-numpy-and-tensorflow-cc275c2b14dd
#------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
import pdb
import sys
#------------------------------------------------------------------------------------------------------------------------
# configuration
#------------------------------------------------------------------------------------------------------------------------
inputSize = 5
outputSize = 3
batchSize = 10
testSize = 5
epochs = 400
learningRate = 0.01
seed = 123456

#----------------------------------------------------------------------------------------------------
class Stop (Exception):
    def __init__ (self):
        #sys.tracebacklimit = 0
        print("stopping script")
#----------------------------------------------------------------------------------------------------


#create functions
#Operator example in forward and backward (Mult)

def forwardMult(A,B):
    return np.matmul(A,B) 

def backwardMult(dC,A,B,dA,dB):
    dA += np.matmul(dC,np.matrix.transpose(B))
    dB += np.matmul(np.matrix.transpose(A),dC)
    
#Loss example in forward and backward (RMSE)
def forwardLoss(predictedOutput, realOutput):
    if(predictedOutput.shape == realOutput.shape):
        loss = np.mean(0.5*np.square(predictedOutput - realOutput))       
    else :
        print("Shape of arrays not the same")
        raise Stop()
    return loss

def backwardLoss(predictedOutput, realOutput):
    if(predictedOutput.shape == realOutput.shape):
        deltaOutput = (predictedOutput - realOutput)/predictedOutput.size
    else:
        print("Shape of arrays not the same")
    return deltaOutput

#Optimizer example (SGD)
#def updateweights(W,dW,learningRate):
#    W -= learningRate * dW

def updateweights(oldWeights,dW,learningRate):
    newWeights = oldWeights - learningRate * dW
    return newWeights

#Generation of fake dataset - we generate random inputs and weights and calculate outputs
np.random.seed(seed)
inputArray = np.random.uniform(-5,5,(batchSize,inputSize))
weights = np.random.uniform(-5,5,(inputSize,outputSize))
targetMatrix = np.matmul(inputArray,weights)
#outputArray = np.matmul(inputArray,weights)
#-----------------------------------------------------------------------------------------
# create a 10x5 array with uniformly distributed values in all rows, but 1's in rows 4-6
# we then learn the weights which transform the inputArray into the targetMatrix
#------------------------------------------------
initialGuessMatrix = np.random.uniform(-5,5,(testSize,inputSize))
outputTest = np.matmul(initialGuessMatrix,weights)



# inputArray
# outputArray
# initialGuessMatrix
# outputTest


#initialization of NN by other random weights
nnWeights = np.random.uniform(-3,3,(inputSize,outputSize))
deltaweights = np.zeros((inputSize,outputSize))
deltainput = np.zeros((batchSize,inputSize))
deltaoutput = np.zeros((inputSize,outputSize))

# targetMatrix[4:,] = 0.1

weights; initialGuessMatrix; nnWeights; targetMatrix
pdb.set_trace()


#Comparing the dataset weights 
weights


#with the NN weights
nnWeights


#----------------------------------------------------------------------------------------------------
historyTrain=[] #Used to record the history of loss
historyTest=[]
i = 1
while i <= epochs:
    nnOutput = forwardMult(inputArray,nnWeights)
    lossTrain = forwardLoss(nnOutput,targetMatrix)
    historyTrain.append(lossTrain)
    nnTest = forwardMult(initialGuessMatrix, nnWeights)
    lossTest = forwardLoss(nnTest,outputTest)
    historyTest.append(lossTest)
    if(i % 50 == 0):
        print("Epoch: " + str(i) + " Loss (train): " + 
              "{0:.3f}".format(lossTrain) + " Loss (test): " + "{0:.3f}".format(lossTest))
    deltaoutput = backwardLoss(nnOutput,targetMatrix)
    pdb.set_trace()
    backwardMult(deltaoutput,inputArray,nnWeights,deltainput,deltaweights)  # last 2 args are updated 
    nnWeights = updateweights(nnWeights,deltaweights, learningRate)
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

