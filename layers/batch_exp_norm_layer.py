import warnings
import theano
import numpy

import theano.tensor as T

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions

#-------------------------------------------------------------------------------
# Begin BatchExpNormLayer
class BatchExpNormLayer(Layer):
    def __init__(self):
        super(BatchExpNormLayer, self).__init__()
        self.layerType='BENlayer'
        
    
    def constructLayer(self, inputShape, initParams, name, 
                       act_func=None, alpha=0.1, 
                       lr_scheduler=None, **layerSpecs):
        self.layerName = name
        self.actFunc = act_func
        self.alpha = alpha
        self.params.setLearningRateScheduler(lr_scheduler)
        
        self.norm_axis = (0,) + tuple(range(2, len(inputShape)))
        x_avg_values = None
        beta_values = None
        
        param_shape = list(inputShape)
        for a in self.norm_axis:
            param_shape[a] = 1
        param_shape = tuple(param_shape)
        
        if initParams is not None:
            if ('x_avg' in initParams) and (initParams['x_avg'] is not None):
                x_avg_values = initParams['x_avg']
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for x_avg, will use constant initialization for x_avg")
        
        if x_avg_values is None:
            x_avg_values = numpy.zeros(inputShape[1], dtype='float32').reshape(param_shape)
        x_avg_expr = theano.shared(x_avg_values, name='x_avg')
        
        if initParams is not None:
            if ('beta' in initParams) and (initParams['beta'] is not None):
                beta_values = initParams['beta']
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for beta, will use constant initialization for beta")
        
        if beta_values is None:
            beta_values = numpy.ones(inputShape[1], dtype='float32').reshape(param_shape)
            
        beta_expr = theano.shared(beta_values, name='beta')
        
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
        # x_avg and x_std should never be tuned through grad descent
        if 'tune' not in layerSpecs:
            layerSpecs['tune'] = {'beta':True, 'x_avg':False}
        else:
            layerSpecs['tune']['x_avg'] = False
            
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['beta', 'x_avg'], layerSpecs)
        
        self.params.addParameters(params = {'beta':beta_expr, 'x_avg':x_avg_expr},
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)
        
    def fprop(self, x, mode='train'):
        if mode == 'test':
            # this is for use during test/validation time
            x_avg = self.params.getParameter('x_avg')
        elif mode == 'calculate':
            x_avg = x.mean(self.norm_axis, keepdims=True)
        elif mode == 'train':
            # otherwise calculate the batch mean and std
            x_avg = x.mean(self.norm_axis, keepdims=True)
            
            # the following trick is learend from lasagne implementation
            running_mean = theano.clone(self.params.getParameter('x_avg'), share_inputs=False)
            
            running_mean_udpate = ((1 - self.alpha) * running_mean
                                    +self.alpha * x_avg)

 
            # set a default update for them
            running_mean.default_update = running_mean_udpate
 
            x_avg += 0 * running_mean
        else:
            raise "mode can only take ['train', 'test', 'calculate']"
        
        self.x_avg = x_avg
        x_avg = T.addbroadcast(x_avg, *self.norm_axis)
        beta = T.addbroadcast(self.params.getParameter('beta'), *self.norm_axis)
        
        bn_x = x / (x_avg + 1e-18) * beta
        return bn_x if self.actFunc is None else self.actFunc(bn_x)


# End BatchExpNormLayer
#-------------------------------------------------------------------------------