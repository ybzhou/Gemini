import numpy

# import theano.sandbox.cuda.dnn as cuDNN
import libwrapper as LW

from layer import Layer

__all__ = ["PoolLayer"]
#-------------------------------------------------------------------------------
# Begin PoolLayer

class PoolLayer(Layer):
    def __init__(self):
        super(PoolLayer, self).__init__()
        self.layerType = 'pool'
        
    def constructLayer(self, inputShape, initParams, name, pool_size, pool_stride,
                       act_func=None, pool_mode='max', pad=0, **layerSpecs):
        self.layerName = name
        self.poolSize = pool_size
        self.poolStride = pool_stride
        self.actFunc = act_func
        self.mode = pool_mode
        self.pad = pad
        
        batchSize, nInputChannels, nInputRows, nInputCols = inputShape
        
        self.inputShape = inputShape
        
        # calculate output size
        self.outputShape = (batchSize, nInputChannels,
                           int((nInputRows - self.poolSize+2*pad)//self.poolStride + 1),
                           int((nInputCols - self.poolSize+2*pad)//self.poolStride + 1))
    
    def fprop(self, x):
        pooled_out = LW.pool2d(x=x, 
                               pool_size=(self.poolSize, self.poolSize), 
                               stride=(self.poolStride, self.poolStride), 
                               padding=(self.pad, self.pad), 
                               mode=self.mode)
        pooled_out = pooled_out if self.actFunc is None else self.actFunc(pooled_out)
        return pooled_out
    
# End PoolLayer
#-------------------------------------------------------------------------------