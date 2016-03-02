import abc

__all__ = ["DataLoader"]

class DataLoader(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def get_ntrain_data(self):
        pass
    
    @abc.abstractmethod
    def get_nvalid_data(self):
        pass
    
    @abc.abstractmethod
    def get_ntest_data(self):
        pass
    
    @abc.abstractmethod
    def get_train_data(self):
        pass
    
    @abc.abstractmethod
    def get_valid_data(self):
        pass
    
    @abc.abstractmethod
    def get_test_data(self):
        pass
    
    @abc.abstractmethod
    def get_train_label(self):
        pass
    
    @abc.abstractmethod
    def get_valid_label(self):
        pass
    
    @abc.abstractmethod
    def get_test_label(self):
        pass
    
    @abc.abstractmethod
    def get_data_dims(self):
        pass
    
    @abc.abstractmethod
    def get_data_stats(self):
        pass