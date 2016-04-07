import abc

# import theano.tensor as T
import libwrapper as LW

class Error:
    __metaclass__ = abc.ABCMeta
    
    """Error takes two layer and compute corresponding error, one is regarded as
       reference, and the other as predicted output"""
    @abc.abstractmethod
    def getError(self, target, predict, *args, **kwargs):
        """Subclasses must implement getError, which take two layer outputs
           and returns a scalar error"""
        raise("Unimplemented Error")
    
class BinaryCrossEntropyError(Error):
    """Binary cross entropy error when the target is binary"""
    def getError(self, target, predict):
        """Target neeed to have the same dimensionality as predict"""
        assert target.ndim == predict.ndim, "the output layer need to output %d dimensions, but got %d dimensions" % (target.ndim, predict.ndim)
        error = LW.mean(LW.sum(LW.binary_cross_entropy(predict, target), axis=1))
        return error
    
class ClassificationErrorScalar(Error):
    """classification error when the target is just one scalar representing class"""
    def getError(self, target, predict):
        """Assumes the targetLayer output be a integer vector, starting from 0"""
        assert target.ndim == 1, "targetLayer need to output a vector but got %d dimensions" % target.ndim
        error = LW.mean(LW.neq(LW.argmax(predict, axis=1), target))
        return error
    
class ClassificationError1ofK(Error):
    """classification error when the target is a one of k vector"""
    def getError(self, target, predict):
        """Assumes the targetLayer output be a integer one of k matrix, starting from 0"""
        assert target.ndim == 2, "targetLayer need to output a matrix but got %d dimensions" % target.ndim
        error = LW.mean(LW.neq(LW.argmax(predict, axis=1), LW.argmax(target, axis=1)))
        return error
    
class SumOfSquaredError(Error):
    """Sum of squared error"""
    def getError(self, target, predict):
        return LW.mean(LW.sum(LW.square(target-predict), axis=1))

class BinaryCodeMatchingError(Error):
    """classification error based on exact matching when the target is random binary vector"""
    def getError(self, target, predict):
        pred = LW.round(predict)
        return LW.mean(LW.neq(LW.sum(LW.neq(pred, target),axis=1),0))
    
class BinaryCodeClassificationError(Error):
    """classification error based on minimum hamming distance when the target is random binary vector"""
    def getError(self, target, predict, code):
        import numpy
        match = LW.dot(predict, code.T) - LW.dot(predict, 1-code.T)
        pred = LW.argmax(match, axis=1)
        truth = LW.argmax(LW.dot(target, code.T) - LW.dot(target, 1-code.T), axis=1)
        return LW.mean(LW.neq(pred, truth))
    