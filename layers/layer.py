import abc

from utils.model.parameter import Parameter

__all__ = ["Layer", "BiDirLayer"]

class Layer(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.prevLayer = []
        self.nextLayer = []
        self.inputShape = None
        self.outputShape = None
        self.layerName = None
        self.layerType = None
        self.params = Parameter({})
    
    def getPreviousLayer(self):
        """Return the previous layer list of the current layer"""
        return self.prevLayer
    
    def getNextLayer(self):
        """Return the next layer list of the current layer"""
        return self.nextLayer
    
    def clearPreviousLayer(self):
        """Set the previous layer list of the current layer to empty list"""
        self.prevLayer = []
        
    def clearNextLayer(self):
        """Set the next layer list of the current layer to empty list"""
        self.nextLayer = []
    
    def addPreviousLayer(self, layer):
        """Add a layer to the previous layer list of the current layer"""
        if isinstance(layer, list):
            self.prevLayer.extend(layer)
        else:
            self.prevLayer.append(layer)
        
    def addNextLayer(self, layer):
        """Add a layer to the next layer list of the current layer"""
        if isinstance(layer, list):
            self.nextLayer.extend(layer)
        else:
            self.nextLayer.append(layer)
    
    def setTuableParams(self, tunable):
        """Set tuable parameters according to tunable, where tunable should be 
           a dictionary with key being the name of the parameter and value be 
           boolean that indicate to tune a particular parameter or not"""
        self.params.setTunableParameters(tunable)
            
    def getTuablePrarms(self):
        return self.params.getTunableParameters()
        
    def setRegularizeParams(self, reg):
        """Set parameter regularizations according to reg, where reg should
           be a dictionary with key being the name of the parameter and value
           be the regularizer"""
        self.params.setParamRegularization(reg)
            
    def getParamRegularization(self):
        return self.params.getParamRegularization()
        
    def getTotalNumParams(self):
        """Return the total number of parameters of the layer"""
        return self.params.getTotalNumParams()
        
    def getNumTunableParams(self):
        """Return the number of tunable parameters of the layer"""
        return self.params.getNumTunableParams()
    
    def getAllParamValues(self):
        """Return all model parameters"""
        return self.params.getAllParameters()
    
    def getParams(self):
        return self.params
    
    def getInputShape(self):
        return self.inputShape
    
    def getOutputShape(self):
        return self.outputShape
    
    @abc.abstractmethod
    def fprop(self, x):
        """Subclasses must implement fprop, which take input x and do forward 
           propagation"""
        raise ('Unimplemented Error')

    @abc.abstractmethod
    def constructLayer(self, inputShape, initParams, layerSpecs):
        """Subclasses must implement this to construct the layer, including
           initialize parameters, setup parameters, etc."""
        raise ('Unimplemented Error')
    
class BiDirLayer(Layer):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def bprop(self, x):
        """Subclasses must implement bprop, which take input x and do forward 
           propagation in the backward direction"""
        raise ('Unimplemented Error')