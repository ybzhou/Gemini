import h5py

from data_loader import DataLoader

class UnlabeledLSUNDataLoader(DataLoader):
    def __init__(self, hdf5_file_path, load_dtype='uint8'):
        
        # load hdf5 file
        train_file = h5py.File(hdf5_file_path, mode='r')
        
        self.train_data = train_file['data']
        if 'mean' in train_file:
            self.train_mean = train_file['mean']
        else:
            self.train_mean = None
        if 'stddev' in train_file:
            self.train_stddev = train_file['stddev']
        else:
            self.train_stddev = None

            
    def get_ntrain_data(self):
        # cifar 10 have 50000 training images
        return self.train_data.shape[0]

    def get_nvalid_data(self):
        # cifar 10 have no validation images
        return 0
    
    def get_ntest_data(self):
        # cifar 10 have 10000 testing images
        return 0
    
    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        assert False, 'LSUN have no validation data'
    
    def get_test_data(self):
        assert False, 'LSUN have no test data'
    
    def get_train_label(self):
        assert False, 'LSUN have no label data'
    
    def get_valid_label(self):
        assert False, 'LSUN have no validation data'
    
    def get_test_label(self):
        assert False, 'LSUN have no test label'
    
    def get_data_dims(self):
        return self.train_data.shape[1]
    