from layer import Layer

__all__ = ["ReshapeLayer"]
#-------------------------------------------------------------------------------
# Begin ReshapeLayer

class ReshapeLayer(Layer):
    def __init__(self):
        super(ReshapeLayer, self).__init__()
        self.layerType='reshape'

    def constructLayer(self, inputShape, initParams, name, shape, act_func=None, **layerSpecs):
        self.layerName = name
        self.inputShape = inputShape
        self.outputShape = shape
        self.actFunc = act_func

    def fprop(self, x):
        return x.reshape(self.outputShape) if self.actFunc is None else self.actFunc(x.reshape(self.outputShape))
    
# End ReshapeLayer
#-------------------------------------------------------------------------------