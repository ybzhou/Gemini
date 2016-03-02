import cPickle
import numpy

from data_loader import DataLoader

class MNISTDataLoader(DataLoader):
    def __init__(self, file_path, subtract_mean=False, divide_std=False):
        f = open(file_path, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        self.train_x = numpy.vstack((train_set[0], valid_set[0]))
        self.train_y = numpy.hstack((train_set[1], valid_set[1]))
        
        self.test_x, self.test_y = test_set
        
        
        self.train_x = numpy.asarray(self.train_x, dtype='float32')
        self.test_x = numpy.asarray(self.test_x, dtype='float32')
        self.train_y = numpy.asarray(self.train_y, dtype='int32')
        self.test_y = numpy.asarray(self.test_y, dtype='int32')
        
        if subtract_mean:
            tm = numpy.mean(self.train_x, axis=0, keepdims=True)
            self.train_x -= tm
            self.test_x -= tm
        
        if divide_std:
            ts = numpy.std(self.train_x, axis=0, keepdims=True)+0.001
            self.train_x /= ts
            self.test_x /= ts
            
            
    def get_ntrain_data(self):
        # MNIST have 60000 training images
        return self.train_x.shape[0]

    def get_nvalid_data(self):
        # MNIST have no validation images
        return 0
    
    def get_ntest_data(self):
        # MNIST have 10000 testing images
        return self.test_x.shape[0]
    
    def get_train_data(self):
        return self.train_x

    def get_valid_data(self):
        assert False, 'MNIST have no validation data'
    
    def get_test_data(self):
        return self.test_x
    
    def get_train_label(self):
        return self.train_y
    
    def get_valid_label(self):
        assert False, 'MNIST have no validation data'
    
    def get_test_label(self):
        return self.test_y
    
    def get_data_dims(self):
        return self.train_x.shape[1]