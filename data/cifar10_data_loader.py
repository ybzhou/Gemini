import cPickle
import numpy

from data_loader import DataLoader

class CIFAR10DataLoader(DataLoader):
    def __init__(self, cifar_file_path, load_dtype='uint8', shuffle_data=False):
        self.shuffle_data = shuffle_data
        
        for i in xrange(5):
            f = open('%s/data_batch_%d' % (cifar_file_path, i+1), 'rb')
            data = cPickle.load(f)
            f.close()
            if i == 0:
                self.train_x = data['data']
                self.train_y = data['labels']
            else:
                self.train_x = numpy.vstack((self.train_x, data['data']))
                self.train_y = numpy.hstack((self.train_y, data['labels']))
        
        f = open('%s/test_batch' % cifar_file_path,  'rb')
        data = cPickle.load(f)
        f.close()
        self.test_x = data['data']
        self.test_y = data['labels']
        
        self.train_x = numpy.asarray(self.train_x, dtype=load_dtype)
        self.test_x = numpy.asarray(self.test_x, dtype=load_dtype)
        self.train_y = numpy.asarray(self.train_y, dtype='int32')
        self.test_y = numpy.asarray(self.test_y, dtype='int32')
            
    def get_ntrain_data(self):
        # cifar 10 have 50000 training images
        return self.train_x.shape[0]

    def get_nvalid_data(self):
        # cifar 10 have no validation images
        return 0
    
    def get_ntest_data(self):
        # cifar 10 have 10000 testing images
        return self.test_x.shape[0]
    
    def get_train_data(self):
        return self.train_x

    def get_valid_data(self):
        assert False, 'CIFAR-10 have no validation data'
    
    def get_test_data(self):
        return self.test_x
    
    def get_train_label(self):
        return self.train_y
    
    def get_valid_label(self):
        assert False, 'CIFAR-10 have no validation data'
    
    def get_test_label(self):
        return self.test_y
    
    def get_data_dims(self):
        return self.train_x.shape[1]
    