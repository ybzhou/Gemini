import abc
import warnings
# import theano.tensor as T
import libwrapper as LW

class NormConstraint:
    def __init__(self, norm):
        self.norm = norm
    
    @abc.abstractmethod
    def applyConstraint(self, param):
        raise('Unimplemented Error')

class MaxRowNormConstraint(NormConstraint):
    def applyConstraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        needFlip = False
        if param.ndim == 4: # a hack for conv layer filters
            prevShape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            needFlip = True
        
        if needFlip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
            
        scale = LW.clip(col_norm, 0, self.norm)
        param *= (scale / (1e-7 + col_norm))
        
        if needFlip:
            param = param.reshape(prevShape)
            
        return param
    
class MaxColNormConstraint(NormConstraint):
    def applyConstraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        needFlip = False
        if param.ndim == 4: # a hack for conv layer filters
            prevShape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            needFlip = True
        
        if needFlip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
            
        scale = LW.clip(col_norm, 0, self.norm)
        param *= (scale / (1e-7 + col_norm))
        
        if needFlip:
            param = param.reshape(prevShape)
            
        return param
    
class L2ColNormConstraint(NormConstraint):
    def applyConstraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        needFlip = False
        if param.ndim == 4: # a hack for conv layer filters
            prevShape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            needFlip = True
        
        if needFlip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
            
        param /= (col_norm+1e-7)
        param *= self.norm
        
        if needFlip:
            param = param.reshape(prevShape)
                        
        return param