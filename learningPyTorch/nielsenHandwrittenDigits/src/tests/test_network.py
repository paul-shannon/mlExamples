# coding: utf-8
import sys
sys.path.append("..")
#----------------------------------------------------------------------------------------------------
import network
import mnist_loader
import pdb
#----------------------------------------------------------------------------------------------------
def runTests():
    test_constructor()

#----------------------------------------------------------------------------------------------------
def test_constructor():

    print("--- test_constructor")

    numpy.random.seed(17)  # for reproducability

    layerCounts = [5, 10, 2]  # number of neurons in each of the 
    net = network.Network(layerCounts)

    assert(net.num_layers == 3)
    assert(net.sizes == layerCounts)

    assert(isinstance(net.biases, list))
    assert(len(net.biases) == 2)           # no biases in the first (input) layer
    assert(net.biases[0].shape == (10,1))  # element 0 describes hidden layer 
    assert(net.biases[1].shape == (2,1))   # element 1 describes the output layer 

        # every node in each non-input layer has as many weights as there are nodes
        # in the preceeding layer. thus
        #   all 10 hidden layer nodes have 5 weights
        #   both output layer nodes have 10 weights
    assert(isinstance(net.weights, list))
    assert(len(net.weights) == 2)          # no biases in the first (input) layer
    assert(net.weights[0].shape == (10,5)) # element 0 describes hidden layer 
    assert(net.weights[1].shape == (2,10)) # element 0 describes hidden layer 

   
    pdb.set_trace()
    #assert(net.biases = [np.random.randn(y, 1) for y in sizes[1:]]
     #   self.weights = [np.random.randn(y, x)
     # for x, y in zip(sizes[:-1], sizes[1:])]


#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    runTests()

