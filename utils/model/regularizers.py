import abc
# import theano.tensor as T
import libwrapper as LW

class Regularizer:
    __metaclass__ = abc.ABCMeta
    def __init__(self, coef):
        self.coef = coef
    
    @abc.abstractmethod
    def getRegularization(self, param):
        raise('Unimplemented Error')
    
    def getRegularizationCoef(self):
        return self.coef
    
    def setRegularizationCoef(self, coef):
        self.coef = coef
    
class L2Regularizer(Regularizer):
    def getRegularization(self, param):
        return self.coef*LW.sum(LW.square(param))
    
class L1Regularizer(Regularizer):
    def getRegularization(self, param):
        return self.coef*LW.sum(LW.abs(param))