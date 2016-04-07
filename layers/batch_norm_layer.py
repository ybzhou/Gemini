import warnings
# import theano
import numpy

# import theano.tensor as T
import libwrapper as LW

from layer import Layer
from utils.model.layer_utils import setupDefaultLayerOptions

__all__ = ["BatchNormLayer"]

#-------------------------------------------------------------------------------
# Begin BatchNormLayer
class BatchNormLayer(Layer):
    def __init__(self):
        super(BatchNormLayer, self).__init__()
        self.layerType='BNlayer'
        
    
    def constructLayer(self, inputShape, initParams, name, 
                       act_func=None, alpha=0.1, lr_scheduler=None, **layerSpecs):
        self.layerName = name
        self.actFunc = act_func
        self.alpha = alpha
        self.params.setLearningRateScheduler(lr_scheduler)
        
        self.norm_axis = (0,) + tuple(range(2, len(inputShape)))
        x_avg_values = None
        x_std_values = None
        beta_values = None
        gamma_values = None
        
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
        x_avg_expr = LW.data_variable(value=x_avg_values, name='x_avg')
            
        if initParams is not None:
            if ('x_std' in initParams) and (initParams['x_std'] is not None):
                x_std_values = initParams['x_std']
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for x_std, will use constant initialization for x_std")
        
        if x_std_values is None:
            x_std_values = numpy.ones(inputShape[1], dtype='float32').reshape(param_shape) #here
            
        x_std_expr = LW.data_variable(value=x_std_values, name='x_std')
        
        if initParams is not None:
            if ('beta' in initParams) and (initParams['beta'] is not None):
                beta_values = initParams['beta']
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for beta, will use constant initialization for beta")
        
        if beta_values is None:
            beta_values = numpy.zeros(inputShape[1], dtype='float32').reshape(param_shape)
            
        beta_expr = LW.data_variable(value=beta_values, name='beta')
        
        if initParams is not None:
            if ('gamma' in initParams) and (initParams['gamma'] is not None):
                gamma_values = initParams['gamma']
            else:
                warnings.warn("initParams provided but did not provide actual initialization"
                              +" value for gamma, will use constant initialization for gamma")
        
        if gamma_values is None:
            gamma_values = numpy.ones(inputShape[1], dtype='float32').reshape(param_shape)
            
        gamma_expr = LW.data_variable(value=gamma_values, name='gamma')
        
        self.inputShape = inputShape
        self.outputShape = self.inputShape
        
        # x_avg and x_std should never be tuned through grad descent
        if 'tune' not in layerSpecs:
            layerSpecs['tune'] = {'beta':True, 'gamma':True, 'x_std':False, 'x_avg':False}
        else:
            layerSpecs['tune']['x_std'] = False
            layerSpecs['tune']['x_avg'] = False
            
        tune, reg, constraint, lr, mu = setupDefaultLayerOptions(['beta','gamma', 'x_avg', 'x_std'], layerSpecs)
        
        self.params.addParameters(params = {'beta':beta_expr, 'gamma':gamma_expr, 'x_avg':x_avg_expr, 'x_std':x_std_expr},
                                 tune = tune, 
                                 regularizer=reg, 
                                 constraint=constraint,
                                 learning_rate=lr, 
                                 momentum=mu)
        
    def fprop(self, x, mode='train'):
        if mode == 'test':
            # this is for use during test/validation time
            x_avg = self.params.getParameter('x_avg')
            x_sq_nc = self.params.getParameter('x_std')
        elif mode == 'calculate':
            x_avg = x.mean(self.norm_axis, keepdims=True)
            x_sq_nc = LW.mean(LW.square(x), axis=self.norm_axis, keepdims=True)
        elif mode == 'train':
            x_avg = x.mean(self.norm_axis, keepdims=True)
            x_sq_nc = LW.mean(LW.square(x), axis=self.norm_axis, keepdims=True)
            
            
            running_mean = LW.clone(self.params.getParameter('x_avg'), share_inputs=False)
            running_sq_nc = LW.clone(self.params.getParameter('x_std'), share_inputs=False)

            # the following trick is learend from lasagne implementation
            running_mean_udpate = ((1 - self.alpha) * running_mean
                                    +self.alpha * x_avg)
              
            running_sq_nc_update = ((1 - self.alpha) * running_sq_nc 
                                    + self.alpha * x_sq_nc)
  
            # set a default update for them
            running_mean.default_update = running_mean_udpate
            running_sq_nc.default_update = running_sq_nc_update
  
            x_avg += 0 * running_mean
            x_sq_nc += 0 * running_sq_nc
            
        else:
            raise "mode can only take ['train', 'test', 'calculate']"
        
        self.x_std = x_sq_nc
        self.x_avg = x_avg
        x_std = LW.sqrt(x_sq_nc - LW.square(x_avg) + 1e-8)
        x_avg = LW.addbroadcast(x_avg, *self.norm_axis)
        x_std = LW.addbroadcast(x_std, *self.norm_axis)
        gamma = LW.addbroadcast(self.params.getParameter('gamma'), *self.norm_axis)
        beta = LW.addbroadcast(self.params.getParameter('beta'), *self.norm_axis)
        
        bn_x = (x - x_avg) * (gamma / x_std) + beta 
        return bn_x if self.actFunc is None else self.actFunc(bn_x)


# End BatchNormLayer
#-------------------------------------------------------------------------------