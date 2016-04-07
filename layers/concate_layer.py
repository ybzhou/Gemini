# import theano.tensor as T
import libwrapper as LW
from layer import Layer

__all__ = ["ConcateLayer"]

#-------------------------------------------------------------------------------
# Begin Concatelayer

class ConcateLayer(Layer):
    def __init__(self):
        super(ConcateLayer, self).__init__()
        self.layerType='concatenate'
        
    def constructLayer(self, inputShape, initParams, name, **layerSpecs):
        self.layerName = name
        self.inputShape = [l.getOutputShape() for l in self.prevLayer]
        nout = 0
        for l in self.prevLayer:
            s = l.getOutputShape()
            ndim = len(s)
            nout += s[1]
        
        self.outputShape  = []
        for i in xrange(ndim):
            self.outputShape.append(self.prevLayer[0].getOutputShape()[i])
        
        self.outputShape[1] = nout
        self.outputShape = tuple(self.outputShape)
    
    def fprop(self, x):
        concatenated = LW.concatenate(x, axis=1)
        return concatenated
    
# End Concatelayer
#-------------------------------------------------------------------------------