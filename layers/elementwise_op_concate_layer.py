from layer import Layer

__all__ = ["ElmOpLayer"]

#-------------------------------------------------------------------------------
# Begin ElmOpLayer

class ElmOpLayer(Layer):
    def __init__(self):
        super(ElmOpLayer, self).__init__()
        self.layerType='elmop_concate'
        
    def constructLayer(self, inputShape, initParams, name, operation, **layerSpecs):
        self.layerName = name
        self.inputShape = [l.getOutputShape() for l in self.prevLayer]
        self.operation = operation
        
        assert operation in ['add', 'mul'], "elementwise operation concatenate layer only support 'add' or 'mul' operation" 
        
        assert len(self.inputShape)>1, \
        'need to have at least 2 inputs to elementwise operation concatenate layer, got only %d inputs' % len(self.inputShape)
        
        shape = self.inputShape[0]
        for ss in self.inputShape:
            assert shape == ss, 'all input to elementwise operation concatenate layer need to have the same shape'
                
        self.outputShape = shape
        
        
    
    def fprop(self, x):
        
        ret = x[0]
        for d in x[1:]:
            if self.operation == 'add':
                ret += d
            elif self.operation == 'mul':
                ret *= d
            else:
                raise('undefined operation')
            
        return ret
    
# End ElmOpLayer
#-------------------------------------------------------------------------------