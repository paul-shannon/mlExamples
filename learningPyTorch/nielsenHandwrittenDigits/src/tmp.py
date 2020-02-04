import pickle
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
        #p = u.load()
        #print(p)
training_data, validation_data, test_data = u.load()
        f.close()
