import abc

import theano.tensor as T


__all__ = ['Sigmoid', 'Tanh', 'Rectifier', 'Softmax', 'Softplus']

class Activation:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def act_func(self, x):
        pass
    
    def __call__(self, x):
        return self.act_func(x)

class Sigmoid(Activation):
    
    def act_func(self, x):
        return T.nnet.sigmoid(x)
    
class Tanh(Activation):
    
    def act_func(self, x):
        return T.tanh(x)
    
class Rectifier(Activation):
    
    def act_func(self, x):
        return T.nnet.relu(x)
    
class Softmax(Activation):
    
    def act_func(self, x):
        return T.nnet.softmax(x)
    
class Softplus(Activation):
    
    def act_func(self, x):
        return T.nnet.softplus(x)