import abc

__all__ = ["LabeledDataProvider", "UnlabeledDataProvider"]

class DataProvider():
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, data_loader, 
                 batch_size=100,
                 nvalid_samples=0,
                 is_test=False, 
                 *args, **kwargs):
        
        self.data_loader = data_loader
        self.is_test = is_test
        self.batch_size = batch_size
        self.nvalid_samples = nvalid_samples
    
    @abc.abstractmethod
    def get_number_of_train_batches(self):
        '''
           return the total number of batches for the training data
        '''
        pass
    
    @abc.abstractmethod
    def get_number_of_valid_batches(self):
        '''
           return the total number of batches for the validation data
        '''
        pass
    
    @abc.abstractmethod
    def get_number_of_train_data(self):
        '''
           return the total number of training data
        '''
        pass
    
    @abc.abstractmethod
    def get_number_of_valid_data(self):
        '''
           return the total number of validation data
        '''
        pass

    @abc.abstractmethod
    def get_data_stats(self):
        '''
           return statistics from data such as mean, std, etc..
        '''
        pass
    

class LabeledDataProvider(DataProvider):
    @abc.abstractmethod
    def get_train_labeled_data_and_idx(self, batch_idx):
        '''
           return theano shared variable that contains training data and labels and 
           its corresponding batch_index with in that chunck of data
        '''
        pass
    
    @abc.abstractmethod
    def get_valid_labeled_data_and_idx(self, batch_idx):
        '''
           return theano shared variable that contains validation data and labels and 
           its corresponding batch_index with in that chunck of data
        '''
        pass



class UnlabeledDataProvider(DataProvider):
    @abc.abstractmethod
    def get_train_data_and_idx(self, batch_idx):
        '''
           return theano shared variable that contains training data and labels and 
           its corresponding batch_index with in that chunck of data
        '''
        pass
    
    @abc.abstractmethod
    def get_valid_data_and_idx(self, batch_idx):
        '''
           return theano shared variable that contains validation data and labels and 
           its corresponding batch_index with in that chunck of data
        '''
        pass
