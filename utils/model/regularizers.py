import abc
import theano.tensor as T

class Regularizer:
    def __init__(self, coef):
        self.coef = coef
    
    @abc.abstractmethod
    def getRegularization(self, param):
        raise('Unimplemented Error')
    
class L2Regularizer(Regularizer):
    def getRegularization(self, param):
        return self.coef*T.sum(T.sqr(param))
    
class L1Regularizer(Regularizer):
    def getRegularization(self, param):
        return self.coef*T.sum(T.abs_(param))