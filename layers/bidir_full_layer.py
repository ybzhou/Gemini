import warnings
import theano
import numpy

import theano.tensor as T

from layer import BiDirLayer
from utils.model.layer_utils import setupDefaultLayerOptions

__all__ = ["BidirFullLayer"]

#-------------------------------------------------------------------------------
# Begin BidirFullLayer
 
class BidirFullLayer(BiDirLayer):
    def __init__(self):
        super(BidirFullLayer, self).__init__()
        self.layerType='bi_full'
 
    def constructLayer(self, inputShape, initParams, name, w_init, hiddens, 
                       b_hid_init=0, b_vis_init=0, fact_func=None, bact_func=None,
                       lr_scheduler=None, tie_weights=False, **layerSpecs):
         
        self.layerName = name
        self.wInit = w_init
        self.bHidInit = b_hid_init
        self.bVisInit = b_vis_init
        self.forwardActFunc = fact_func
        self.backwardActFunc = bact_func
        self.params.setLearningRateScheduler(lr_scheduler)
        self.inputShape = inputShape
        self.tieWeights = tie_weights
        
        nHiddenSize = hiddens
         
        if len(inputShape) > 2:
            inputShape = (inputShape[0], numpy.prod(inputShape[1:]))

        self.outputShape = (inputShape[0], nHiddenSize)
         
        W_values = None
        W_prime_values = None
        b_values = None
        b_prime_values = None
        if initParams is not None:
            if ('W' in initParams) and (initParams['W'] is not None):
                W_values = initParams['W']
                assert len(W_values.shape) == 2, \
                       "Initialize W dimension does not match, expected to be a" \
                       +"2 dimenional matrix got %d dimensions" % len(W_values.shape)
                assert W_values.shape == (inputShape[-1], self.outputShape[-1]), \
                       ("Initialize W shape is incorrect, expected: (%d, %d), got: (%d, %d)" \
                        % (inputShape[-1], self.outputShape[-1], W_values.shape[0], W_values.shape[1]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for W, will use random initialization for W")
                
        if initParams is not None:
            if not self.tieWeights:
                if ('W_prime' in initParams) and (initParams['W_prime'] is not None):
                    W_prime_values = initParams['W_prime']
                    assert len(W_prime_values.shape) == 2, \
                           "Initialize W_prime dimension does not match, expected to be a" \
                           +"2 dimenional matrix got %d dimensions" % len(W_prime_values.shape)
                    assert W_prime_values.shape == (self.outputShape[-1], inputShape[-1]), \
                           ("Initialize W_prime shape is incorrect, expected: (%d, %d), got: (%d, %d)" \
                            % (self.outputShape[-1], inputShape[-1], W_prime_values.shape[0], W_prime_values.shape[1]))
                else:
                    warnings.warn("initParams provided but did not provide actual initialization"
                                  +" value for W, will use random initialization for W")
         
        if initParams is not None:
            if ('b' in initParams) and (initParams['b'] is not None):
                b_values = initParams['b']
                assert len(b_values.shape) == 1, \
                       "Initialize b dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(b_values.shape)
                assert b_values.shape == (self.outputShape[-1],), \
                       ("Initialize b shape is incorrect, expected: %d, got: %d" \
                        % (self.outputShape[-1], b_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for b, will use constant initialization for b")
                
        if initParams is not None:
            if ('b_prime' in initParams) and (initParams['b_prime'] is not None):
                b_prime_values = initParams['b_prime']
                assert len(b_values.shape) == 1, \
                       "Initialize b_prime dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(b_prime_values.shape)
                assert b_prime_values.shape == (inputShape[-1],), \
                       ("Initialize b_prime shape is incorrect, expected: %d, got: %d" \
                        % (inputShape[-1], b_prime_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for b_prime, will use constant initialization for b_prime")
         
        if W_values is None:
            W_values = self.wInit.init(inputShape[-1], self.outputShape[-1])
 
        W_expr = theano.shared(name='W', value=W_values, borrow=True)
        
        if self.tieWeights:
            W_prime_expr = W_expr.T
        else:
            if W_prime_values is None:
                W_prime_values = self.wInit.init(self.outputShape[-1], inputShape[-1])
            W_prime_expr = theano.shared(name='W_prime', value=W_prime_values, borrow=True)
        
        if b_values is None:
            b_values = self.bHidInit*numpy.ones((self.outputShape[-1],), dtype='float32')
             
        b_expr = theano.shared(name='b', value=b_values, borrow=True)
        
        if b_prime_values is None:
            b_prime_values = self.bVisInit*numpy.ones((inputShape[-1],), dtype='float32')
             
        b_prime_expr = theano.shared(name='b_prime', value=b_prime_values, borrow=True)
         
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['W','W_prime','b', 'b_prime'], layerSpecs)
        
        # no need to tune W_prime if it is tied with W
        if self.tieWeights:
            params = {'W':W_expr, 'b':b_expr, 'b_prime':b_prime_expr}
        else:
            params = {'W':W_expr, 'W_prime':W_prime_expr, 'b':b_expr, 'b_prime':b_prime_expr}
        
        self.params.addParameters(params = params,
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)

    def fprop(self, x):
        if x.ndim > 2:
            x = x.flatten(2)
        pre_act = T.dot(x, self.params.getParameter('W')) + self.params.getParameter('b')
        output = pre_act if self.forwardActFunc is None else self.forwardActFunc(pre_act)
        return output
     
    def bprop(self, x):
        if self.tieWeights:
            pre_act = T.dot(x, self.params.getParameter('W').T) + self.params.getParameter('b_prime')
        else:
            pre_act = T.dot(x, self.params.getParameter('W_prime')) + self.params.getParameter('b_prime')
        output = pre_act if self.backwardActFunc is None else self.backwardActFunc(pre_act)
        
        if len(self.inputShape) != 2:
            output = output.reshape(self.inputShape)
        
        return output

# End BidirFullLayer
#-------------------------------------------------------------------------------