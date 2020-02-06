# coding: utf-8
import sys
sys.path.append("..")
#----------------------------------------------------------------------------------------------------
import network
import mnist_loader
import numpy as np
import pdb
import matplotlib.pyplot as plt   # optionally used to display the handwritten character images
#----------------------------------------------------------------------------------------------------
def runTests():

    test_mnist_load_data()
    test_mnist_load_data_wrapper()

    test_constructor()

    test_sigmoid()
    test_sigmoid_prime()
    test_feedforward()
    # test_update_mini_batch()
    test_backprop()
    test_SGD()
    test_evaluate()
    test_cost_derivative()
    test_fullRun()


#----------------------------------------------------------------------------------------------------
"""
The training_data is returned as a tuple with two entries.
The first entry contains the actual training images.  This is a
numpy ndarray with 50,000 entries.  Each entry is, in turn, a
numpy ndarray with 784 values, representing the 28 * 28 = 784
pixels in a single MNIST image.

The second entry in the training_data tuple is a numpy ndarray
containing 50,000 entries.  Those entries are just the digit
values (0...9) for the corresponding images contained in the first
entry of the tuple.

The validation_data and test_data are similar, except
each contains only 10,000 images.

This is a nice data format, but for use in neural networks it's
helpful to modify the format of the training_data a little.
That's done in the wrapper function load_data_wrapper(), see
below.

from load_data_wrapper:
In particular, ``training_data`` is a list containing 50,000
2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
containing the input image.  ``y`` is a 10-dimensional
numpy.ndarray representing the unit vector corresponding to the
correct digit for ``x``.

``validation_data`` and ``test_data`` are lists containing 10,000
2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
numpy.ndarry containing the input image, and ``y`` is the
corresponding classification, i.e., the digit values (integers)
corresponding to ``x``.

Obviously, this means we're using slightly different formats for
the training data and the validation / test data.  These formats
turn out to be the most convenient for use in our neural network
code.
"""

def test_mnist_load_data():

    print("--- test_mnist_load_data")

    training_data, validation_data, test_data = mnist_loader.load_data()

    assert(isinstance(training_data, tuple))
    assert(training_data[0].shape == (50000, 784))
    assert(training_data[1].shape == (50000,))

    assert(isinstance(validation_data, tuple))
    assert(validation_data[0].shape == (10000, 784))
    assert(validation_data[1].shape == (10000,))

    assert(isinstance(test_data, tuple))

       #-------------------------------------------------------------
       # look at a sample image from the training data: a bitmap of 4
       #-------------------------------------------------------------
    assert(training_data[1][2] == 4)
    img1_bits  = training_data[0][2].reshape(28, 28)
    # plt.imshow(img1_bits); plt.show()

       #-------------------------------------------------------------
       # now an image from the validation data
       #-------------------------------------------------------------
    assert(validation_data[1][0] == 3)
    # plt.imshow(validation_data[0][0].reshape(28, 28)); plt.show()

       #-------------------------------------------------------------
       # and the test_data
       #-------------------------------------------------------------
    assert(test_data[1][0] == 7)
    # plt.imshow(test_data[0][0].reshape(28, 28)); plt.show()

#------------------------------------------------------------------------------------------------------------------------
# the wrapper version zips each image & letter name into tuples, apparently for ease of processing
# make sure we got that right.
def test_mnist_load_data_wrapper():

    print("--- test_mnist_load_data_wrapper")

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    isinstance(training_data, list)
    assert(len(training_data) == 50000)
    assert(isinstance(training_data[0], tuple))

    imageNumber = 12
    #plt.imshow(training_data[imageNumber][0].reshape(28, 28)); plt.show()
    assert(training_data[imageNumber][1][3][0] == 1)   
    
    #----------------------------------------------------------------------------------------------------
    # old-style, the load_data version of these data structures, with
    # element 0 being an array of 28x28 image vectors, elemen 1 being an array of simple integers
    #
    # assert(training_data[1][2] == 4)
    # img1_bits  = training_data[0][2].reshape(28, 28)
    # plt.imshow(img1_bits); plt.show()
    #----------------------------------------------------------------------------------------------------

    isinstance(validation_data, list)
    assert(len(validation_data) == 10000)

    #----------------------------------------------------------------------------------------------------
    # note that the second item in each tuple is not the array used by training_data, but instead
    # a simple integer.  mnist_loader.py says:
    #
    #   Obviously, this means we're using slightly different formats for
    #   the training data and the validation / test data.  These formats
    #   turn out to be the most convenient for use in our neural network  code.
    #
    # when I understand this I will return and add more notes here
    #----------------------------------------------------------------------------------------------------
    imageNumber = 12
    #plt.imshow(validation_data[imageNumber][0].reshape(28, 28)); plt.show()
    assert(validation_data[imageNumber][1] == 3)

    isinstance(test_data, list)
    assert(len(test_data) == 10000)
    #plt.imshow(test_data[imageNumber][0].reshape(28, 28)); plt.show()
    assert(test_data[imageNumber][1] == 9)
    
#------------------------------------------------------------------------------------------------------------------------
def test_constructor():

    print("--- test_constructor")

    np.random.seed(17)  # for reproducability

    layerCounts = [5, 10, 2]  # number of neurons in each of the 
    net = network.Network(layerCounts)

    assert(net.num_layers == 3)
    assert(net.sizes == layerCounts)

    assert(isinstance(net.biases, list))
    assert(len(net.biases) == 2)           # no biases in the first (input) layer
    assert(net.biases[0].shape == (10,1))  # element 0 describes hidden layer 
    assert(net.biases[1].shape == (2,1))   # element 1 describes the output layer 

        # every node in each layer (except the input layer)  has as many weights as 
        # there are nodes in the preceeding layer. thus, with the requested layer
        # counts [5, 10, 2]   all 10 hidden layer nodes have 5 weights
        # both  of the 2 output layer nodes have 10 weights

    assert(isinstance(net.weights, list))
    assert(len(net.weights) == 2)          # no biases in the first (input) layer
    assert(net.weights[0].shape == (10,5)) # element 0 describes hidden layer 
    assert(net.weights[1].shape == (2,10)) # element 0 describes hidden layer 

   
    #assert(net.biases = [np.random.randn(y, 1) for y in sizes[1:]]
     #   self.weights = [np.random.randn(y, x)
     # for x, y in zip(sizes[:-1], sizes[1:])]


#----------------------------------------------------------------------------------------------------
def test_sigmoid():

    print("--- test_sigmoid")

    z = 0.5
    zSigmoid = network.sigmoid(z)
    expectedValue = 0.6224593312018546
    assert(zSigmoid - expectedValue < 0.000001)  # allow for small numerical variations

    
#----------------------------------------------------------------------------------------------------
def test_sigmoid_prime():

    print("--- test_sigmoid_prime")

    z = 0.5
    zSigmoidPrime = network.sigmoid_prime(z)
    expectedValue = 0.2350037122015945
    assert(zSigmoidPrime - expectedValue < 0.000001)  # allow for small numerical variations

    
#----------------------------------------------------------------------------------------------------
def test_feedforward():

    print("--- test_feedforward")

    
#----------------------------------------------------------------------------------------------------
def test_update_mini_batch():

    print("--- test_update_mini_batch")
   
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    n = len(training_data)
    mini_batch_size = 10
    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

    layerCounts = [5, 10, 2]  # number of neurons in each of the 
    net = network.Network(layerCounts)

    learningRate = 3
    net.update_mini_batch(mini_batches, learningRate)
    pdb.set_trace()

    
#----------------------------------------------------------------------------------------------------
def test_backprop():

    print("--- test_backprop")

    
#----------------------------------------------------------------------------------------------------
def test_SGD():

    print("--- test_SGD")

    
#----------------------------------------------------------------------------------------------------
def test_evaluate():

    print("--- test_evaluate")

    
#----------------------------------------------------------------------------------------------------
def test_cost_derivative():

    print("--- test_cost_derivative")

    
#----------------------------------------------------------------------------------------------------
def test_fullRun():

    print("--- test_fullRun")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0)

    # net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    
#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    runTests()

