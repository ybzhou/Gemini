from layer import BiDirLayer
from utils.model.layer_utils import corrupt

__all__ = ["BidirNoiseLayer"]

#-------------------------------------------------------------------------------
# Begin BidirNoiseLayer
class BidirNoiseLayer(BiDirLayer):
    def __init__(self):
        super(BidirNoiseLayer, self).__init__()
        self.layerType='bi_noise'
    
    def constructLayer(self, inputShape, initParams, name, fnoise_type, bnoise_type,
                       fnoise_level, bnoise_level, theano_rng, **layerSpecs):
        self.layerName = name
        self.forwardNoiseType = fnoise_type
        self.forwardNoiseLevel = fnoise_level
        self.backwardNoiseType = bnoise_type
        self.backwardNoiseLevel = bnoise_level
        self.theanoRng = theano_rng
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
    def fprop(self, x):
        return corrupt(x, self.theanoRng, self.forwardNoiseType, self.forwardNoiseLevel)
    
    def bprop(self, x):
        return corrupt(x, self.theanoRng, self.backwardNoiseType, self.backwardNoiseLevel)

# End BidirNoiseLayer
#-------------------------------------------------------------------------------