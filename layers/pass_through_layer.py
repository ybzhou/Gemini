from layer import Layer

__all__ = ["PassThroughLayer"]

#-------------------------------------------------------------------------------
# Begin PassThroughLayer
class PassThroughLayer(Layer):
    def __init__(self):
        super(PassThroughLayer, self).__init__()
        self.layerType='pass'
    
    def constructLayer(self, inputShape, initParams, name, act_func=None, **layerSpecs):
        self.layerName = name
        self.actFunc = act_func
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
    def fprop(self, x):
        return x if self.actFunc is None else self.actFunc(x)

# End PassThroughLayer
#-------------------------------------------------------------------------------