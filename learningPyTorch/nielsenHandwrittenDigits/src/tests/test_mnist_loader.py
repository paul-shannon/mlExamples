import pdb
import sys
sys.path.append("..")
import mnist_loader
#------------------------------------------------------------------------------------------------------------------------
def runTests():

    test_readPickle()
    
#------------------------------------------------------------------------------------------------------------------------
# lots of things were needed here, primarily due to changes from python 2 to 3
def test_readPickle():

    print("--- test_readPickle")
    training_data, validation_data, test_data = mnist_loader.load_data()

    assert(isinstance(training_data, tuple))
    assert(training_data[0].shape == (50000, 784))
    assert(training_data[1].shape == (50000,))

    assert(isinstance(validation_data, tuple))
    assert(validation_data[0].shape == (10000, 784))
    assert(validation_data[1].shape == (10000,))

    assert(isinstance(test_data, tuple))

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    isinstance(training_data, list)
    assert(len(training_data) == 50000)

    isinstance(validation_data, list)
    assert(len(validation_data) == 10000)

    isinstance(test_data, list)
    assert(len(test_data) == 10000)
    
    
#------------------------------------------------------------------------------------------------------------------------

