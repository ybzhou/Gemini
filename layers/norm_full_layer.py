import warnings
# import theano
import numpy

# import theano.tensor as T
import libwrapper as LW

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions

__all__ = ['NormFullLayer']

#-------------------------------------------------------------------------------
# Begin NormFullLayer

_sub_const = numpy.cast['float32'](numpy.sqrt(2/numpy.pi)*0.5)#numpy.cast['float32'](numpy.sqrt(0.5)*0.5)#numpy.float32(numpy.sqrt(2*numpy.pi/(numpy.pi-1))*numpy.sqrt(2/numpy.pi)*0.5)

class NormFullLayer(Layer):
    def __init__(self):
        super(NormFullLayer, self).__init__()
        self.layerType='norm_full'
        
    def constructLayer(self, inputShape, initParams, name, w_init, hiddens, 
                       b_init=0,
                       act_func=None, lr_scheduler=None, W_expr=None,
                       post_act_normalize=True,
                       **layerSpecs):
        self.layerName = name
        self.wInit = w_init
        self.bInit = b_init
        self.actFunc = act_func
        self.post_act_normalize = post_act_normalize
        
        self.params.setLearningRateScheduler(lr_scheduler)
        
        nHiddenSize = hiddens
        
        if len(inputShape) > 2:
            self.inputShape = (inputShape[0], numpy.prod(inputShape[1:]))
        else:
            self.inputShape = inputShape
            
        self.outputShape = (inputShape[0], nHiddenSize)
        
        W_values = None
        b_values = None
        gamma_values = None
        
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
                       "Initialize b dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(b_values.shape)
                assert b_values.shape == (self.outputShape[-1],), \
                       ("Initialize b shape is incorrect, expected: %d, got: %d" \
                        % (self.outputShape[-1], b_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for b, will use constant initialization for b")
        
        if initParams is not None:
            if ('gamma' in initParams) and (initParams['gamma'] is not None):
                gamma_values = initParams['gamma']
                assert len(gamma_values.shape) == 1, \
                       "Initialize gamma dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(gamma_values.shape)
                assert gamma_values.shape == (self.outputShape[-1],), \
                       ("Initialize gamma shape is incorrect, expected: %d, got: %d" \
                        % (self.outputShape[-1], gamma_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for gamma, will use constant initialization for gamma")
        
        if gamma_values is None:
            gamma_values = numpy.ones((self.outputShape[1],), dtype='float32')
            
        gamma_expr = LW.data_variable(value=gamma_values, name='gamma')
        
        if W_expr is not None:
            W_expr = W_expr
        else:
            if W_values is None:
                W_values = self.wInit.init(self.inputShape[-1], self.outputShape[-1])
            W_expr = LW.data_variable(name='W', value=W_values)
        
        if b_values is None:
            b_values = self.bInit*numpy.ones((self.outputShape[-1],), dtype='float32')
            
        b_expr = LW.data_variable(name='b', value=b_values)
        
#         beta_expr = theano.shared(name='beta', value=numpy.ones(self.outputShape[-1], dtype='float32'))
        
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['W','b', 'gamma'], layerSpecs)
        self.params.addParameters(params = {'W':W_expr, 'b':b_expr, 'gamma':gamma_expr},
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)
        
    def fprop(self, x):
        if x.ndim > 2:
            x = x.flatten(2)
        
        W = self.params.getParameter('W')
        gamma = self.params.getParameter('gamma')
        b = self.params.getParameter('b')
        W_non_center_std = LW.sqrt(LW.sum(LW.square(W), axis=0))*numpy.float32(1.21)
        pre_act = gamma*LW.dot(x, W) / W_non_center_std + b
        output = self.actFunc(pre_act) if self.actFunc is not None else pre_act
        if self.post_act_normalize:
            output = (output-_sub_const)/numpy.float32(numpy.sqrt(0.5-0.5/numpy.pi))
        return output


# End NormFullLayer
#-------------------------------------------------------------------------------