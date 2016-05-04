import warnings
# import theano
import numpy

# import theano.tensor as T
# import theano.sandbox.cuda.dnn as cuDNN
import libwrapper as LW

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions, corrupt

__all__ = ["NormConvLayer"]

#-------------------------------------------------------------------------------
# Begin NormConvLayer
_sub_const = numpy.cast['float32'](numpy.sqrt(2/numpy.pi)*0.5)#numpy.float32(numpy.sqrt(0.5)*0.5)## #numpy.float32(numpy.sqrt(2*numpy.pi/(numpy.pi-1))*numpy.sqrt(2/numpy.pi)*0.5)

class NormConvLayer(Layer):
    """Convolutional layer"""
    def __init__(self):
        super(NormConvLayer, self).__init__()
        self.layerType = 'norm_conv'
    
    def constructLayer(self, inputShape, initParams, name, batch_size, w_init,
                       channels, filter_size, 
                       strid_size=1, pad=0, b_init=0, act_func=None, 
                       lr_scheduler=None, 
                       post_act_normalize=True, W_expr=None, **layerSpecs):
        self.layerName = name
        self.batchSize = batch_size
        self.strideSize = strid_size
        self.nPad = pad
        self.wInit = w_init
        self.bInit = b_init
        self.actFunc = act_func
        self.post_act_normalize = post_act_normalize
        self.params.setLearningRateScheduler(lr_scheduler)
        
        nFilters = channels
        filterSize = filter_size
        
        self.filterShape = (nFilters, inputShape[1], filterSize, filterSize)
        self.inputShape = inputShape
        
        # calculate output size
        self.outputShape = (self.batchSize, nFilters,
                           int((inputShape[2] - filterSize + self.nPad*2)//self.strideSize + 1),
                           int((inputShape[3] - filterSize + self.nPad*2)//self.strideSize + 1))
        
        
        W_values = None
        b_values = None
        gamma_values = None
        if initParams is not None:
            if ('W' in initParams) and (initParams['W'] is not None):
                W_values = initParams['W']
                assert len(W_values.shape) == 4, \
                       "Initialize W dimension does not match, expected to be a" \
                       +"4 dimenional tensor got %d dimensions" % len(W_values.shape)
                assert W_values.shape == self.filterShape, \
                       ("Initialize W shape is incorrect, expected: (%d, %d, %d, %d), got: (%d, %d, %d, %d)"\
                        % (self.filterShape[0], self.filterShape[1], self.filterShape[2], self.filterShape[3],\
                           W_values.shape[0], W_values.shape[1], W_values.shape[2], W_values.shape[3]))
                
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for W, will use random initialization for W")
        
        if initParams is not None:
            if ('b' in initParams) and (initParams['b'] is not None):
                b_values = initParams['b']
                assert len(b_values.shape) == 1, \
                       "Initialize W dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(b_values.shape)
                assert b_values.shape == (self.filterShape[0],), \
                       ("Initialize b shape is incorrect, expected: %d, got: %d" \
                        % (self.filterShape[0], b_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for b, will use random initialization for b")
    
        if initParams is not None:
            if ('gamma' in initParams) and (initParams['gamma'] is not None):
                gamma_values = initParams['gamma']
                assert len(gamma_values.shape) == 1, \
                       "Initialize gamma dimension does not match, expected to be a" \
                       +"vector got %d dimensions" % len(gamma_values.shape)
                assert gamma_values.shape == (self.filterShape[0],), \
                       ("Initialize gamma shape is incorrect, expected: %d, got: %d" \
                        % (self.filterShape[0], gamma_values.shape[0]))
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for gamma, will use constant initialization for gamma")

        if gamma_values is None:
            gamma_values = numpy.ones((self.filterShape[0],), dtype='float32')
            
        gamma_expr = LW.data_variable(value=gamma_values, name='gamma')
        
        if W_expr is not None:
            W_expr = W_expr
        else:
            if W_values is None:
                W_values = self.wInit.init(numpy.prod(self.filterShape[1:]), self.filterShape[0], 
                                           numpy.prod(self.filterShape[1:]), numpy.prod(self.filterShape)/self.filterShape[1])
            W_expr = LW.data_variable(name='W', value=W_values.reshape(self.filterShape))
        
        if b_values is None:
            b_values = self.bInit*numpy.ones((self.filterShape[0],), dtype='float32')
            
        b_expr = LW.data_variable(name='b', value=b_values)
        
#         beta_expr = theano.shared(name='beta', value=numpy.ones(self.filterShape[0], dtype='float32'))
        
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['W','b', 'gamma'], layerSpecs)
        
        self.params.addParameters(params = {'W':W_expr, 'b':b_expr, 'gamma':gamma_expr},
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)
    
    def getInputShape(self):
        return self.inputShape
    
    def getOutputShape(self):
        return self.outputShape
    
    def fprop(self, x):
        W = self.params.getParameter('W')
        b = self.params.getParameter('b')
        gamma = self.params.getParameter('gamma')
        # convolve input feature maps with filters
        conv_out = LW.conv2d(x=x,
                             filter=W,
                             border_mode=(self.nPad, self.nPad),
                             stride=(self.strideSize, self.strideSize))

        W_non_center_std = LW.sqrt(LW.sum(LW.square(W), axis=(1,2,3)))*numpy.float32(1.21)
        conv_out /=  LW.dimshuffle(W_non_center_std, 'x', 0, 'x', 'x')
        conv_out *= LW.dimshuffle(gamma, 'x', 0, 'x', 'x')
        conv_out +=  LW.dimshuffle(b, 'x', 0, 'x', 'x')
        
        if self.post_act_normalize and self.actFunc:
            conv_out = (self.actFunc(conv_out) - _sub_const)/numpy.float32(numpy.sqrt(0.5-0.5/numpy.pi))
        
        return conv_out
    
# End NormConvLayer
#-------------------------------------------------------------------------------