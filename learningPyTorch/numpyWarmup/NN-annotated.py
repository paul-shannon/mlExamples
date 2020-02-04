# coding: utf-8
#------------------------------------------------------------------------------------------------------------------------
# from https://medium.com/datathings/a-neural-network-fully-coded-in-numpy-and-tensorflow-cc275c2b14dd
#------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pdb
# import matplotlib.pyplot as plt 

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
    new_dA = dA + np.matmul(dC,np.matrix.transpose(B))
    new_dB = dB + np.matmul(np.matrix.transpose(A),dC)
    return (new_dA, new_dB)
    
#Loss example in forward and backward (RMSE)
def forwardLoss(predictedOutput,realOutput):
    if(predictedOutput.shape == realOutput.shape):
        loss = np.mean( 0.5*np.square(predictedOutput - realOutput))
    else :
        print("Shape of arrays not the same")
    return loss

def backwardLoss(predictedOutput, realOutput):
    if(predictedOutput.shape == realOutput.shape):
        loss = (predictedOutput - realOutput)/predictedOutput.size
    else :
        print("Shape of arrays not the same")
    return loss

# SGD: stochastic gradient descent
def updateWeights(W,dW,learningRate):
    newWeights = W - learningRate * dW
    return newWeights

#Generation of fake dataset - we generate random inputs and weights and calculate outputs
np.random.seed(seed)
RO_starterMatrix = np.random.uniform(-5,5,(batchSize,inputSize))
weights = np.random.uniform(-5,5,(inputSize,outputSize))
RO_targetMatrix = np.matmul(RO_starterMatrix,weights)
inputTest = np.random.uniform(-5,5,(testSize,inputSize))
outputTest = np.matmul(inputTest,weights)

RO_starterMatrix
RO_targetMatrix
inputTest
outputTest


#initialization of NN by other random weights
nnWeights = np.random.uniform(-3,3,(inputSize,outputSize))
deltaWeights = np.zeros((inputSize,outputSize))
deltainput = np.zeros((batchSize,inputSize))
deltaOutput = np.zeros((inputSize,outputSize))


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

      #---------------------------------------------------------------------------------------------------
      # forward step: calculate a new 10x3 matrix from current weights and 10x5 read-only starter matrix
      # calculate error: RMS of (current predictedOutput - RO_targetMatrix)
      #---------------------------------------------------------------------------------------------------

    predictedOutput = forwardMult(RO_starterMatrix, nnWeights) # just np.matmul(RO_starterMatrix, nmWeights) 
    lossTrain = forwardLoss(predictedOutput, RO_targetMatrix)  # np.mean(0.5*np.square(predictedOutput - RO_targetMatrix))
    historyTrain.append(lossTrain)
    pdb.set_trace()

      #------------------------- 
      #  Forward pass test
      #------------------------- 
    nnTest = forwardMult(inputTest, nnWeights)
    lossTest = forwardLoss(nnTest, outputTest)
    historyTest.append(lossTest)
    #Print Loss every 50 epochs: 
    if(i%10==0):
        print("Epoch: " + str(i) + " Loss (train): " + "{0:.3f}".format(lossTrain) + " Loss (test): " + "{0:.3f}".format(lossTest))
        
      #--------------------------------
      # Backpropagate: update weights
      #-------------------------------

          # backward loss = (predictedOutput - realOutput)/predictedOutput.size
          # def backwardMult(dC,A,B,dA,dB):
          #     new_dA = dA + np.matmul(dC,np.matrix.transpose(B))
          #     new_dB = dB + np.matmul(np.matrix.transpose(A),dC)
          #     return (new_dA, new_dB)

    deltaOutput = backwardLoss(predictedOutput, RO_targetMatrix)
    (deltainput, deltaWeights) = backwardMult(deltaOutput,RO_starterMatrix,nnWeights,deltainput, deltaWeights)
    
    #Apply optimizer
    nnWeights = updateWeights(nnWeights, deltaWeights, learningRate)
    
    #Reset deltas 
    deltainput = np.zeros((batchSize,inputSize))
    deltaWeights = np.zeros((inputSize,outputSize))
    deltaOutput = np.zeros((inputSize,outputSize))
    
    #Start new epoch
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

