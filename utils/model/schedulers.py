import abc
import theano

import numpy
import theano.tensor as T

from collections import OrderedDict
from theano.ifelse import ifelse

class Scheduler(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, **kargs):
        self.args = kargs
        self.rate = None
    
    @abc.abstractmethod
    def get_updates(self):
        return

class AnnealScheduler(Scheduler):
    def __init__(self, update_freq, anneal_coef, min_rate, **kargs):
        super(AnnealScheduler,self).__init__(**kargs)
        self.update_freq = numpy.cast['int32'](update_freq)
        self.anneal_coef = numpy.cast['float32'](anneal_coef)
        self.min_rate = numpy.cast['float32'](min_rate)
        
    def get_updates(self, rate):
        iters = theano.shared(numpy.cast['int32'](1))
        self.rate = theano.shared(numpy.cast['float32'](rate))
        updated_lr = T.maximum(self.min_rate, self.rate * self.anneal_coef)
        updates = OrderedDict()
        updates[iters] =  (iters+numpy.cast['int32'](1))%self.update_freq
        updates[self.rate] = ifelse(T.gt(iters%self.update_freq, 0), self.rate, updated_lr)
        
        return updates
    
class ExponentialDecayScheduler(Scheduler):
    def __init__(self, start_lr, final_lr, total_epochs, update_freq, **kargs):
        super(ExponentialDecayScheduler, self).__init__(**kargs)
        self.decay = numpy.cast['float32']((final_lr / start_lr)**(1./total_epochs))
#         self.decay = numpy.cast['float32']((start_lr - final_lr)**(1./total_epochs))
        self.update_freq = numpy.cast['int32'](update_freq)
        
    def get_updates(self, rate):
        iters = theano.shared(numpy.cast['int32'](1))
        self.rate = theano.shared(numpy.cast['float32'](rate))
        updated_lr = self.rate * self.decay
        updates = OrderedDict()
        updates[iters] =  (iters+numpy.cast['int32'](1))%self.update_freq
        updates[self.rate] = ifelse(T.gt(iters%self.update_freq, 0), self.rate, updated_lr)
        return updates