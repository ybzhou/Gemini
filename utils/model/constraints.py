import abc
import warnings
import libwrapper as LW

class NormConstraint:
    def __init__(self, norm):
        self.norm = norm
    
    @abc.abstractmethod
    def apply_constraint(self, param):
        raise('Unimplemented Error')

class MaxRowNormConstraint(NormConstraint):
    def apply_constraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        need_flip = False
        if param.ndim == 4: # a hack for conv layer filters
            prev_shape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            need_flip = True
        
        if need_flip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
            
        scale = LW.clip(col_norm, 0, self.norm)
        param *= (scale / (1e-7 + col_norm))
        
        if need_flip:
            param = param.reshape(prev_shape)
            
        return param
    
class MaxColNormConstraint(NormConstraint):
    def apply_constraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        need_flip = False
        if param.ndim == 4: # a hack for conv layer filters
            prev_shape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            need_flip = True
        
        if need_flip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
            
        scale = LW.clip(col_norm, 0, self.norm)
        param *= (scale / (1e-7 + col_norm))
        
        if need_flip:
            param = param.reshape(prev_shape)
            
        return param
    
class L2ColNormConstraint(NormConstraint):
    def apply_constraint(self, param):
        if param.ndim != 4 and param.ndim != 2:
            warnings.warn("Norm constraints are normally applied to matrices"
                          +" or 4-dimensional tensors, but currently got "
                          +"%d dimensions, please make sure this is the desired"
                          +" parameter to apply norm constraints" % param.ndim)
            
        need_flip = False
        if param.ndim == 4: # a hack for conv layer filters
            prev_shape = param.shape
            # conv layer filter shape is (nChannelOut, nChannelIn, r, c)
            param = param.flatten(2)
            # now it is (nout, nin), which is different from (nin, nout) 
            # from fulling connected networks, so need to flip here
            need_flip = True
        
        if need_flip:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=1, keepdims=True))
        else:
            col_norm = LW.sqrt(LW.sum(LW.square(param), axis=0, keepdims=True))
            
        param /= (col_norm+1e-7)
        param *= self.norm
        
        if need_flip:
            param = param.reshape(prev_shape)
                        
        return param