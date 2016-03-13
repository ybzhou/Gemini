import warnings
import theano
import numpy

import theano.tensor as T

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions


__all__ = ["FullLayer"]

#-------------------------------------------------------------------------------
# Begin FullLayer

class FullLayer(Layer):
    """Fully connected layer"""
    def __init__(self):
        super(FullLayer, self).__init__()
        self.layerType='full'

    def constructLayer(self, inputShape, initParams, name, w_init, hiddens, 
                       b_init=0, act_func=None, lr_scheduler=None, weights_outside=None,
                       ignore_bias=False,
                       **layerSpecs):
        self.layerName = name
        self.wInit = w_init
        self.bInit = b_init
        self.actFunc = act_func
        self.ignore_bias = ignore_bias
        self.params.setLearningRateScheduler(lr_scheduler)
        self.weights_outside = weights_outside
        
        nHiddenSize = hiddens
        
        if len(inputShape) > 2:
            self.inputShape = (inputShape[0], numpy.prod(inputShape[1:]))
        else:
            self.inputShape = inputShape
        self.outputShape = (inputShape[0], nHiddenSize)
        
        W_values = None
        b_values = None
        if initParams is not None:
            if ('W' in initParams) and (initParams['W'] is not None):
                W_values = initParams['W']
                assert len(W_values.shape) == 2, \
                       "Initialize W dimension does not match, expected to be a" \
                       +"2 dimenional matrix got %d dimensions" % len(W_values.shape)
                assert W_values.shape == (self.inputShape[-1], self.outputShape[-1]), \
                       ("Initialize W shape is incorrect, expected: (%d, %d), got: (%d, %d)" \
                        % (self.inputShape[-1], self.outputShape[-1], W_values.shape[0], W_values.shape[1]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for W, will use random initialization for W")
        
        if initParams is not None:
            if ('b' in initParams) and (initParams['b'] is not None):
                b_values = initParams['b']
                assert len(b_values.shape) == 1, \
                       "Initialize W dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(b_values.shape)
                assert b_values.shape == (self.outputShape[-1],), \
                       ("Initialize b shape is incorrect, expected: %d, got: %d" \
                        % (self.outputShape[-1], b_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for b, will use constant initialization for b")
        
        
        
        if weights_outside is not None:
            W_expr = weights_outside[0]
        else:
            if W_values is None:
                W_values = self.wInit.init(self.inputShape[-1], self.outputShape[-1])
            W_expr = theano.shared(name='W', value=W_values, borrow=True)
        
        if b_values is None:
            b_values = self.bInit*numpy.ones((self.outputShape[-1],), dtype='float32')
            
        b_expr = theano.shared(name='b', value=b_values, borrow=True)
        
        if self.ignore_bias:
            if 'tune' in layerSpecs:
                layerSpecs['tune']['b'] = False
            else:
                layerSpecs['tune'] = {'W':True, 'b':False}
            
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['W','b'], layerSpecs)
        self.params.addParameters(params = {'W':W_expr, 'b':b_expr},
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)
    
    def fprop(self, x):
        if x.ndim > 2:
            x = x.flatten(2)
        
        if self.weights_outside is None or self.weights_outside[1]==False:
            W = self.params.getParameter('W')
        else:
            W = self.params.getParameter('W').T
        
        if self.ignore_bias:
            pre_act = T.dot(x, W)
        else:
            pre_act = T.dot(x, W) + self.params.getParameter('b')
        output = pre_act if self.actFunc is None else self.actFunc(pre_act)
        return output
    
# End FullLayer
#-------------------------------------------------------------------------------