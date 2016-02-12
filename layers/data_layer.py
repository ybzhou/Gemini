from layer import Layer

__all__ = ["DataLayer"]

#-------------------------------------------------------------------------------
# Begin DataLayer

class DataLayer(Layer):
    def __init__(self):
        super(DataLayer, self).__init__()
        self.layerType='data'
    
    def constructLayer(self, inputShape, initParams, name, input_type, 
                       data_index=0, **kwargs):
        self.layerName = name
        self.inputShape = inputShape
        self.inputType = input_type
        self.outputShape = self.inputShape
        self.data_index = data_index
    
    def fprop(self, x):
        if len(self.inputShape) != 2:
            # x suppose to be a matrix
            # need to reshape here
            return x.reshape(self.inputShape)
        else:
            return x
    
    def getPreviousLayer(self):
        raise ('Data layer does not take any input from previous layer')
    
    def addPreviousLayer(self, layer):
        raise ('Data layer does not take any input from previous layer')
    
# End DataLayer
#-------------------------------------------------------------------------------