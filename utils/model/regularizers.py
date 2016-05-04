import abc
import libwrapper as LW

class Regularizer:
    __metaclass__ = abc.ABCMeta
    def __init__(self, coef):
        self.coef = coef
    
    @abc.abstractmethod
    def get_regularization(self, param):
        raise('Unimplemented Error')
    
    def get_regularization_coef(self):
        return self.coef
    
    def set_regularization_coef(self, coef):
        self.coef = coef
    
class L2Regularizer(Regularizer):
    def get_regularization(self, param):
        return self.coef*LW.sum(LW.square(param))
    
class L1Regularizer(Regularizer):
    def get_regularization(self, param):
        return self.coef*LW.sum(LW.abs(param))