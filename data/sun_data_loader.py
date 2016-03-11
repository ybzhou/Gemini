import numpy
import warnings

from data_loader import DataLoader

class SUNDataLoader(DataLoader):
    def __init__(self, file_path, load_dtype='uint8', norm_range=None,
                 subtract_mean=False, divide_std=False, contrast_norm=False):
        
        self.data = numpy.load(file_path)
        
        self.data = numpy.asarray(self.data, dtype=load_dtype)

        if subtract_mean or divide_std or contrast_norm or norm_range is not None:
            if load_dtype != 'float32':
                warnings.warn('loading data with float32 format instead of %s,' % load_dtype
                              +'because subtract_mean or divide_std requires float32 data')
            self.data = numpy.asarray(self.data, dtype='float32')
            
        if subtract_mean:
            tm = numpy.mean(self.train_x, axis=0, keepdims=True)
            self.data -= tm
            
        if divide_std:
            ts = numpy.std(self.train_x, axis=0, keepdims=True)+0.001
            self.data /= ts
            
        if norm_range is not None:
            assert len(norm_range)==2, 'norm_range is required to have two elements'
            self.data -= self.data.min(axis=1, keepdims=True)
            self.data /= (self.data.max(axis=1, keepdims=True)-self.data.min(axis=1, keepdims=True))
            self.data *= norm_range[1]-norm_range[0]
            self.data += norm_range[0]
            
        if contrast_norm:
            self.data -= self.data.mean(axis=1, keepdims=True)
            
            # truncate to 3 std
            tstd = 3*self.data.std(axis=1, keepdims=True)
            self.data = numpy.maximum(numpy.minimum(self.data, tstd), -tstd)/tstd
            
            # scale to -0.9 0.9
            self.data *= 0.9
            
            
    def get_ntrain_data(self):
        return self.data.shape[0]

    def get_nvalid_data(self):
        return 0
    
    def get_ntest_data(self):
        return 0
    
    def get_train_data(self):
        return self.data

    def get_valid_data(self):
        assert False, 'SUN have no validation data'
    
    def get_test_data(self):
        assert False, 'SUN does not have test data'
    
    def get_train_label(self):
        assert False, 'SUN does not have data labels'
    
    def get_valid_label(self):
        assert False, 'SUN have no validation data'
    
    def get_test_label(self):
        assert False, 'SUN does not have label'
    
    def get_data_dims(self):
        return self.data.shape[1]
    
    def get_data_stats(self):
        return self.data.mean(axis=0)
    