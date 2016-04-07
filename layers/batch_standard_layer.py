from layer import Layer

import libwrapper as LW

__all__ = ["BatchStandardizeLayer"]

#-------------------------------------------------------------------------------
# Begin BatchNormLayer
class BatchStandardizeLayer(Layer):
    def __init__(self):
        super(BatchStandardizeLayer, self).__init__()
        self.layerType='BStandardLayer'
        
    
    def constructLayer(self, inputShape, initParams, name, 
                       **layerSpecs):
        self.layerName = name
        
        self.inputShape = inputShape
        self.outputShape = inputShape
        
    def fprop(self, x, isTest=False):
        
        ret = x
        if not isTest:
            norm_axis = (1,)+tuple(range(2,len(self.inputShape)))
            x_avg = LW.mean(x, axis=norm_axis, keepdims=True)
            x_std = LW.std(x, axis=norm_axis, keepdims=True)
            ret = (x-x_avg)/(x_std+1e-4)
            
        return ret


# End BatchNormLayer
#-------------------------------------------------------------------------------