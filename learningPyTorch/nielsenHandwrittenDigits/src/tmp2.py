import mnist_loader
import network
import pickle
import gzip

f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
training_data, validation_data, test_data = u.load()
f.close()

net = network.Network([784, 30, 10])
training_data
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
