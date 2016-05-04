import numpy
# import theano

from numpy import linalg
import libwrapper as LW


class SameValueInit:
    def __init__(self, num):
        self.num = num
    
    def init(self, n_in, n_out, *args, **kwargs):
        W = numpy.ones((n_in, n_out), dtype=LW.FLOAT_TYPE)*self.num
        
        return numpy.asarray(W, dtype=LW.FLOAT_TYPE)

class OrthogonalWeightInit:
    def __init__(self, numpy_rng):
        self.rng = numpy_rng
    
    def init(self, n_in, n_out):
        W = self.rng.randn(n_in, n_out, *args, **kwargs)
        trans = False
        if n_in < n_out:
            W = W.T
            trans = True
        W, _ = linalg.qr(W)
        ret_W = W.T if trans else W
        return numpy.asarray(ret_W, dtype=LW.FLOAT_TYPE)
    
class GaussianWeightInit:
    def __init__(self, numpy_rng, sigma):
        self.rng = numpy_rng
        self.sigma = sigma
        
    def init(self, num_vis, num_hid, *args, **kwargs):
        W = numpy.asarray(self.sigma*self.rng.randn(num_vis, num_hid), 
                          dtype=LW.FLOAT_TYPE)
        
        return W

class NormalizedWeightInit:
    def __init__(self, numpy_rng):
        self.rng = numpy_rng
        
    def init(self, num_vis, num_hid, fan_in=None, fan_out=None, *args, **kwargs):
        if fan_in is None:
            fan_in = num_vis
        if fan_out is None:
            fan_out = num_hid
            
        W = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / (fan_in + fan_out)),
            high=numpy.sqrt(6. / (fan_in + fan_out)),
#             low=-numpy.sqrt(3. / (num_vis)),
#             high=numpy.sqrt(3. / (num_vis)),
            size=(num_vis, num_hid)), dtype=LW.FLOAT_TYPE)
        
        return W