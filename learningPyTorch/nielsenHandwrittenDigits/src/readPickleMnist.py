# coding: utf-8
import pickle
import gzip
file = "../data/mnist.pkl.gz"
# with open('mnist.pkl', 'rb') as f:

with gzip.open(file, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
#    u = pickle._Unpickler(f)
#    u.encoding = 'latin1'
#    p = u.load()
    
