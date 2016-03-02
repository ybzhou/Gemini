import cPickle
import numpy
import warnings

from data_loader import DataLoader

class CIFAR10DataLoader(DataLoader):
    def __init__(self, cifar_file_path, load_dtype='uint8', norm_range=None,
                 subtract_mean=False, divide_std=False, contrast_norm=False):
        
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
        
        if subtract_mean or divide_std or contrast_norm or norm_range is not None:
            if load_dtype != 'float32':
                warnings.warn('loading data with float32 format instead of %s,' % load_dtype
                              +'because subtract_mean or divide_std requires float32 data')
            self.train_x = numpy.asarray(self.train_x, dtype='float32')
            self.test_x = numpy.asarray(self.test_x, dtype='float32')
            
        if subtract_mean:
            tm = numpy.mean(self.train_x, axis=0, keepdims=True)
            self.train_x -= tm
            self.test_x -= tm
            
        if divide_std:
            ts = numpy.std(self.train_x, axis=0, keepdims=True)+0.001
            self.train_x /= ts
            self.test_x /= ts
            
        if norm_range is not None:
            assert len(norm_range)==2, 'norm_range is required to have two elements'
            self.train_x -= self.train_x.min(axis=1, keepdims=True)
            self.test_x -= self.test_x.min(axis=1, keepdims=True)
            self.train_x /= (self.train_x.max(axis=1, keepdims=True)-self.train_x.min(axis=1, keepdims=True))
            self.test_x /= (self.test_x.max(axis=1, keepdims=True)-self.test_x.min(axis=1, keepdims=True))
            self.train_x *= norm_range[1]-norm_range[0]
            self.test_x *= norm_range[1]-norm_range[0]
            self.train_x += norm_range[0]
            self.test_x += norm_range[0]
            
        if contrast_norm:
            self.train_x -= self.train_x.mean(axis=1, keepdims=True)
            self.test_x -= self.test_x.mean(axis=1, keepdims=True)
            
            # truncate to 3 std
            tstd = 3*self.train_x.std(axis=1, keepdims=True)
            self.train_x = numpy.maximum(numpy.minimum(self.train_x, tstd), -tstd)/tstd
            tstd = 3*self.test_x.std(axis=1, keepdims=True)
            self.test_x = numpy.maximum(numpy.minimum(self.test_x, tstd), -tstd)/tstd
            
            # scale to -0.9 0.9
            self.train_x *= 0.9
            self.test_x *= 0.9
            
            
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
    
    def get_data_stats(self):
        return self.train_x.mean(axis=0)
    