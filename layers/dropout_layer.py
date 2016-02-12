from layer import Layer
from utils.model.layer_utils import corrupt

__all__ = ["DropoutLayer"]

#-------------------------------------------------------------------------------
# Begin DropoutLayer
class DropoutLayer(Layer):
    def __init__(self):
        super(DropoutLayer, self).__init__()
        self.layerType='dropout'
    
    def constructLayer(self, inputShape, initParams, name, 
                       theano_rng, noise_level=0.5, **layerSpecs):
        self.layerName = name
        self.noiseLevel = noise_level
        self.theanoRng = theano_rng
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
    def fprop(self, x, isTest=False):
        if isTest:
            return x*(1-self.noiseLevel)
        else:
            return corrupt(x, self.theanoRng, 'mask', self.noiseLevel)

# End DropoutLayer
#-------------------------------------------------------------------------------