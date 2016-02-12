from layer import Layer
from utils.model.layer_utils import corrupt

__all__ = ["NoiseLayer"]

#-------------------------------------------------------------------------------
# Begin NoiseLayer
class NoiseLayer(Layer):
    def __init__(self):
        super(NoiseLayer, self).__init__()
        self.layerType='noise'
    
    def constructLayer(self, inputShape, initParams, name, noise_type, noise_level, 
                       theano_rng, **layerSpecs):
        self.layerName = name
        self.noiseType = noise_type
        self.noiseLevel = noise_level
        self.theanoRng = theano_rng
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
    def fprop(self, x):
        return corrupt(x, self.theanoRng, self.noiseType, self.noiseLevel)

# End NoiseLayer
#-------------------------------------------------------------------------------