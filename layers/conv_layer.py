import warnings
import theano
import numpy

import theano.sandbox.cuda.dnn as cuDNN

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions, corrupt

__all__ = ["ConvLayer"]

#-------------------------------------------------------------------------------
# Begin ConvLayer

class ConvLayer(Layer):
    """Convolutional layer"""
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.layerType = 'conv'
    
    def constructLayer(self, inputShape, initParams, name, batch_size, w_init, 
                       channels, filter_size, strid_size=1, pad=0, b_init=0, 
                       act_func=None, lr_scheduler=None, weigth_outside=None, ignore_bias=False, 
                       algo='small',
                       **layerSpecs):
        self.layerName = name
        self.batchSize = batch_size
        self.strideSize = strid_size
        self.nPad = pad
        self.wInit = w_init
        self.bInit = b_init
        self.actFunc = act_func
        self.ignore_bias = ignore_bias
        self.algo = algo
        self.weigth_outside = weigth_outside
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
    
        
        
        if weigth_outside is not None:
            W_expr = weigth_outside[0]
        else:
            if W_values is None:
                W_values = self.wInit.init(numpy.prod(self.filterShape[1:]), self.filterShape[0], 
                                           numpy.prod(self.filterShape[1:]), numpy.prod(self.filterShape)/self.filterShape[1])
            W_expr = theano.shared(name='W', value=W_values.reshape(self.filterShape), borrow=True)
        
        if b_values is None:
            b_values = self.bInit*numpy.ones((self.filterShape[0],), dtype='float32')
            
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
    
    def getInputShape(self):
        return self.inputShape
    
    def getOutputShape(self):
        return self.outputShape
    
    def fprop(self, x):
        # convolve input feature maps with filters
        if self.weigth_outside is None or self.weigth_outside[1]==False:
            W = self.params.getParameter('W')
        else:
            W = self.params.getParameter('W').dimshuffle(1,0,2,3)
            
        conv_out = cuDNN.dnn_conv(
                    img=x, 
                    kerns=W, 
                    border_mode=(self.nPad, self.nPad),
                    subsample=(self.strideSize, self.strideSize),
                    algo=self.algo)
        
        if not self.ignore_bias:
            conv_out +=  self.params.getParameter('b').dimshuffle('x', 0, 'x', 'x')
        conv_out = conv_out if self.actFunc is None else self.actFunc(conv_out)

        return conv_out
    
# End ConvLayer
#-------------------------------------------------------------------------------