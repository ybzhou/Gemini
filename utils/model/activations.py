import abc

import theano.tensor as T


__all__ = ['Sigmoid', 'Tanh', 'Rectifier', 'Softmax', 'Softplus', 'LeakyRelu']

class Activation:
    __metaclass__ = abc.ABCMeta
    name = ''
    @abc.abstractmethod
    def act_func(self, x):
        pass
    
    def __call__(self, x):
        return self.act_func(x)

class Sigmoid(Activation):
    name = 'sigmoid'
    def act_func(self, x):
        return T.nnet.sigmoid(x)
    
class Tanh(Activation):
    name = 'tanh'
    def act_func(self, x):
        return T.tanh(x)
    
class Rectifier(Activation):
    name = 'relu'
    def act_func(self, x):
        return T.nnet.relu(x)
    
class Softmax(Activation):
    name = 'softmax'
    def act_func(self, x):
        return T.nnet.softmax(x)
    
class Softplus(Activation):
    name = 'softplus'
    def act_func(self, x):
        return T.nnet.softplus(x)
    
class LeakyRelu(Activation):
    def __init__(self, slope):
        self.name = 'LeakyRelu'
        self.slope = slope
    
    def act_func(self, x):
        pos = 0.5*(1+self.slope)
        neg = 0.5*(1-self.slope)
        return pos*x + neg*T.abs_(x)