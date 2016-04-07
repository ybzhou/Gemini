import abc
# import theano
import numpy

# import theano.tensor as T
import libwrapper as LW

from collections import OrderedDict


class Optimizer(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, **kargs):
        self.args = kargs
    
    @abc.abstractmethod
    def get_updates(self, cost, params):
        return
    
################################################################################
# Begin SGD_Optimizer

class SGD_Optimizer(Optimizer):
    
    def get_updates(self, cost, params):
        updates = OrderedDict()
        for param in params:
            lr_scheduler = param.getLearningRateScheduler()
            tunableParamNames = param.getTunableParameternames()
            for pn in tunableParamNames:
                # Gradient update
                # 1. get lr for that parameter
                # 2. get parameter value
                # 3. get gradient
                # 4. do gradient descent
                # 5. apply parameter specific constraint
                # 6. add updated parameter to parameter update
                # 7. add learning rate update according to the scheduler
                lr = param.getLearningRate(pn)
                if lr_scheduler is not None:
                    updates.update(lr_scheduler.get_updates(lr))
                    lr = lr_scheduler.rate
                    
                crtParam = param.getParameter(pn)
                crtParamGrad = LW.grad(cost, crtParam)
                updatedParam = crtParam - lr * crtParamGrad
                updatedParam = param.applyParamConstraint(pn, updatedParam)
                updates[crtParam] = updatedParam

        return updates
# End SGD_Optimizer
################################################################################

################################################################################
# Begin SGD_Momentum_Optimizer
    
class SGD_Momentum_Optimizer(Optimizer):
    def __init__(self, **kargs):
        super(SGD_Momentum_Optimizer,self).__init__(**kargs)
        
    def get_updates(self, cost, params):
        
        updates = OrderedDict()
        for param in params:
            lr_scheduler = param.getLearningRateScheduler()
            mu_scheduler = param.getMomentumScheduler()
            tunableParamNames = param.getTunableParameternames()
            for pn in tunableParamNames:
                # Gradient update
                # 1. get lr for that parameter and mu for that parameter
                # 2. add learning rate and momentum coefficient update according to the scheduler
                # 3. get parameter value
                # 4. get gradient
                # 5. calculate momentum
                # 6. do gradient descent based on momentum
                # 7. apply parameter specific constraint
                # 8. add updated parameter to parameter update
                
                lr = param.getLearningRate(pn)
                mu = param.getMomentum(pn)
                
                val = numpy.zeros(param.getParameter(pn).get_value().shape, dtype='float32')
                momentum = LW.data_variable(val)
                
                if lr_scheduler is not None:
                    updates.update(lr_scheduler.get_updates(lr))
                    lr = lr_scheduler.rate
                if mu_scheduler is not None:
                    updates.update(mu_scheduler.get_updates(mu))
                    mu = mu_scheduler.rate
                    
                crtParam = param.getParameter(pn)
                crtParamGrad = LW.grad(cost, crtParam)
                
                delta = mu*momentum - lr*crtParamGrad
                updatedParam = crtParam + delta
                updatedParam = param.applyParamConstraint(pn, updatedParam)
                updates[crtParam] = updatedParam
                updates[momentum] = delta
        
        return updates
    
# End SGD_Momentum_Optimizer
################################################################################

################################################################################
# Begin SGD_Nesterov_Optimizer

class SGD_Nesterov_Optimizer(Optimizer):
    def __init__(self, **kargs):
        super(SGD_Nesterov_Optimizer, self).__init__(**kargs)
        
    def get_updates(self, cost, params):
                
        updates = OrderedDict()
        for param in params:
            lr_scheduler = param.getLearningRateScheduler()
            mu_scheduler = param.getMomentumScheduler()
            tunableParamNames = param.getTunableParameternames()
            for pn in tunableParamNames:
                # Gradient update
                # 1. get lr for that parameter and mu for that parameter
                # 2. add learning rate and momentum coefficient update according to the scheduler
                # 3. get parameter value
                # 4. get gradient
                # 5. calculate momentum
                # 6. do gradient descent based on momentum
                # 7. apply parameter specific constraint
                # 8. add updated parameter to parameter update
                
                lr = param.getLearningRate(pn)
                mu = param.getMomentum(pn)
                
                val = numpy.zeros(param.getParameter(pn).get_value().shape, dtype='float32')
                momentum = LW.data_variable(val)
                
                if lr_scheduler is not None:
                    updates.update(lr_scheduler.get_updates(lr))
                    lr = lr_scheduler.rate
                if mu_scheduler is not None:
                    updates.update(mu_scheduler.get_updates(mu))
                    mu = mu_scheduler.rate
                    
                crtParam = param.getParameter(pn)
                crtParamGrad = LW.grad(cost, crtParam)
                
                updated_vel = mu*momentum - lr*crtParamGrad
                inc = mu*updated_vel - lr*crtParamGrad
                
                updatedParam = crtParam + inc
                updatedParam = param.applyParamConstraint(pn, updatedParam)
                updates[crtParam] = updatedParam
                updates[momentum] = updated_vel
                

        return updates
    
# End SGD_Nesterov_Optimizer
################################################################################

################################################################################
# Begin SGD_Rms_Optimizer

class SGD_Rms_Optimizer(Optimizer):
    def __init__(self, decay=0.9, **kargs):
        super(SGD_Rms_Optimizer, self).__init__(**kargs)
        self.gamma = decay

    def get_updates(self, cost, params):
        updates = OrderedDict()
        for param in params:
            lr_scheduler = param.getLearningRateScheduler()
            tunableParamNames = param.getTunableParameternames()
            for pn in tunableParamNames:
                # Gradient update
                # 1. get lr for that parameter
                # 2. add learning rate update according to the scheduler
                # 3. get parameter value
                # 4. get gradient
                # 5. calculate r value
                # 6. do gradient descent based on r values
                # 7. apply parameter specific constraint
                # 8. add updated parameter to parameter update
                
                lr = param.getLearningRate(pn)
                val = numpy.zeros(param.getParameter(pn).get_value().shape, dtype='float32')
                r = LW.data_variable(val)
                if lr_scheduler is not None:
                    updates.update(lr_scheduler.get_updates(lr))
                    lr = lr_scheduler.rate
                    
                crtParam = param.getParameter(pn)
                crtParamGrad = LW.grad(cost, crtParam)
                
                updated_r = (1-self.gamma)*LW.square(crtParamGrad) + self.gamma*r + 1e-6 
                inc = - lr*crtParamGrad/LW.sqrt(updated_r)
                
                updatedParam = crtParam + inc
                updatedParam = param.applyParamConstraint(pn, updatedParam)
                updates[crtParam] = updatedParam
                updates[r] = updated_r
        
        return updates
    
# End SGD_Rms_Optimizer
################################################################################

################################################################################
# Begin SGD_Adam_Optimizer
'''
This implements the Adam optimization method from D. P. Kingma and J. L. Ba
paper title "Adam: A method for stochastic optimization", ICLR 2015
link: http://arxiv.org/pdf/1412.6980.pdf
'''
class SGD_Adam_Optimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, **kargs):
        super(SGD_Adam_Optimizer, self).__init__(**kargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, cost, params):
        t = LW.data_variable(numpy.asarray(0, dtype='float32'))
        updated_t = t + 1
        
        updates = OrderedDict()
        for param in params:
            lr_scheduler = param.getLearningRateScheduler()
            tunableParamNames = param.getTunableParameternames()
            for pn in tunableParamNames:
                crtParam = param.getParameter(pn)
                crtParamGrad = LW.grad(cost, crtParam)
                
                param_shape = param.getParameter(pn).get_value().shape
                m = LW.data_variable(numpy.zeros(param_shape, dtype='float32'))
                v = LW.data_variable(numpy.zeros(param_shape, dtype='float32'))
                
                updated_m = self.beta1 * m + (1-self.beta1) * crtParamGrad
                updated_v = self.beta2 * v + (1-self.beta2) * LW.square(crtParamGrad)
                
                lr = param.getLearningRate(pn)
                if lr_scheduler is not None:
                    updates.update(lr_scheduler.get_updates(lr))
                    lr = lr_scheduler.rate
                
                updated_lr = lr * LW.sqrt(1-LW.pow(self.beta2, updated_t))/(1-LW.pow(self.beta1, updated_t))
                
                
                inc = - updated_lr*updated_m/(LW.sqrt(updated_v) + self.epsilon)
                
                updatedParam = crtParam + inc
                updatedParam = param.applyParamConstraint(pn, updatedParam)
                updates[crtParam] = updatedParam
                updates[m] = updated_m
                updates[v] = updated_v
        
        updates[t] = updated_t
        return updates
    
# End SGD_Adam_Optimizer
################################################################################