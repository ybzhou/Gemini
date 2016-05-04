import abc
import numpy

# import theano.tensor as T
import libwrapper as LW

class Cost:
    __metaclass__ = abc.ABCMeta
    
    """Cost takes two layer and compute corresponding cost, one is regarded as
       reference, and the other as predicted output"""
    @abc.abstractmethod
    def getCost(self, target, predict):
        """Subclasses must implement getCost, which take two layer outputs
           and returns a scalar cost"""
        raise("Unimplemented Error")
    
class SoftmaxCost(Cost):
    """Softmax cost"""
    def getCost(self, target, predict):
        """Assumes the targetLayer output be a integer vector, starting from 0"""
        assert target.ndim == 1, "targetLayer need to output a vector but got %d dimensions" % target.ndim
        cost = -LW.mean(LW.log(predict[LW.arange(target.shape[0]), LW.cast(target, 'int32')]))
        return cost

class CrossEntrypyCost(Cost):
    def getCost(self, target, predict):
        assert target.ndim == 1, "targetLayer need to output a vector but got %d dimensions" % target.ndim
        cost = LW.mean(-predict[:, LW.cast(target, 'int32')] + LW.log(LW.sum(LW.exp(predict), axis=1)))
        return cost
    
class BinaryCrossEntropyCost(Cost):
    """Binary cross entropy cost"""
    def getCost(self, target, predict):
        return LW.mean(LW.sum(LW.binary_cross_entropy(predict, target), axis=1))

class SumOfSquaredCost(Cost):
    """Sum of squared cost"""
    def getCost(self, target, predict):
        naxis = predict.ndim
        sum_axis = numpy.arange(1, naxis)
        return LW.mean(LW.sum(LW.square(target-predict), axis=tuple(sum_axis)))
    
class MultiHingeCost(Cost):
    
    
    """Multi-class Hinge Loss"""
    def getCost(self, target, predict):
        import theano.tensor as T
        import warnings
        warnings.warn("MultiHingeCost currently only supports theano")
        """Assumes the targetLayer output be a integer vector, starting from 0"""
        assert target.ndim == 1, ("targetLayer need to output a vector but got %d dimensions" % target.ndim)
        # false-true, 0
        row_idx = T.arange(target.shape[0])
#         diff = predict - predict[row_idx, target].dimshuffle(0,'x')
#         # manually blank out the true term, so that they do not influence the loss
#         diff = T.inc_subtensor(diff[row_idx, target], -1) 
#         cost = T.mean(T.maximum(0, 1 + T.max(diff, axis=1)))
        new_target = T.extra_ops.to_one_hot(target, predict.shape[1], dtype='float32')
#         new_target = 2.*new_target - 1.
        cost = T.mean(T.sqr(T.maximum(0.,1.- (2.*new_target - 1.)*predict)))
        return cost
        
class SquaredMultiHingeCost(Cost):
    """Squared Multi-class Hinge Loss"""
    def getCost(self, target, predict):
        """Assumes the targetLayer output be a integer vector, starting from 0"""
#         assert target.ndim == 1, ("targetLayer need to output a vector but got %d dimensions" % target.ndim)
#         # false-true, 0
#         row_idx = T.arange(target.shape[0])
#         diff = predict - predict[row_idx, target].dimshuffle(0,'x')
#         # manually blank out the true term, so that they do not influence the loss
#         diff = T.inc_subtensor(diff[row_idx, target], -1) 
# #         cost = T.mean(T.sqr(T.maximum(0, 1 + T.max(diff, axis=1))))
#         cost = T.mean(T.sum(T.sqr(T.maximum(0., 1+diff)), axis=1))
        
#         row_idx = T.arange(target.shape[0])
#         new_target = -T.ones_like(predict, dtype='float32')
#         new_target = T.set_subtensor(new_target[row_idx, target], 1)
#         cost = T.mean(T.sqr(T.maximum(0, 1-new_target*predict)))
#         nb_class = predict.shape[1]
#         one_hot = T.cast(T.extra_ops.to_one_hot(target, nb_class), 'float32')
#         one_hot = one_hot*2 - 1
#         cost = T.mean(T.sqr(T.maximum(0.,1.-one_hot*predict)))
        cost = LW.mean(LW.square(LW.maximum(0.,1.-target*predict)))
        return cost
