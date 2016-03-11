import cPickle
import numpy
import warnings
import os
import scipy.io as io
from data_loader import DataLoader

class TFDDataLoader(DataLoader):
    def __init__(self, tfd_file_path, load_dtype='uint8', fold=0):
        
        data = io.loadmat(os.path.join(tfd_file_path, 'TFD_48x48.mat'))
        X = numpy.asarray(data['images'], dtype=load_dtype)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    #     image = PIL.Image.fromarray(tile_raster_images(X=X,
    #         img_shape=(96,96), tile_shape=(10, 10),
    #         tile_spacing=(1, 1)))
    #     image.show()
    #     labels  = data['labs_ex'].flatten()
    #     labeled = labels != -1
    #     unlabeled   =   labels == -1  
    #     train_X =   X[unlabeled]
    #     valid_X =   X[unlabeled][:500] # Stuf
    #     test_X  =   X[labeled]
        train_idx = data['folds'][:,fold] == 0
        valid_idx = data['folds'][:,fold] == 2
        test_idx = data['folds'][:,fold] == 3
        
        self.train_x = X[train_idx]
        self.valid_x = X[valid_idx]
        self.test_x = X[test_idx]
        
        del data

            
            
    def get_ntrain_data(self):
        return self.train_x.shape[0]

    def get_nvalid_data(self):
        return self.valid_x.shape[0]
    
    def get_ntest_data(self):
        return self.test_x.shape[0]
    
    def get_train_data(self):
        return self.train_x

    def get_valid_data(self):
        return self.valid_x
    
    def get_test_data(self):
        return self.test_x
    
    def get_train_label(self):
        assert False, "TFD is UNlabeled"
    
    def get_valid_label(self):
        assert False, "TFD is UNlabeled"
    
    def get_test_label(self):
        assert False, "TFD is UNlabeled"
    
    def get_data_dims(self):
        return self.train_x.shape[1]
    
    def get_data_stats(self):
        return self.train_x.mean(axis=0)
    