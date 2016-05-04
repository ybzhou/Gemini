import abc
import numpy

import libwrapper as LW

from collections import OrderedDict
from .schedulers import Scheduler

class Optimizer(object):
    """
    Abstract base class for optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, learning_rate):
        """
        Constructor 
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        """
        self.lr = learning_rate
        
    @abc.abstractmethod
    def get_updates(self, cost, params):
        return
    
################################################################################
# Begin SGD_Optimizer

class SGD_Optimizer(Optimizer):
    '''
    Stochastic gradient descent optimizer, subclass of Optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    lr_scheduler : None or Scheduler class object
        learning rate scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    '''
    def __init__(self, learning_rate, lr_scheduler=None):
        '''
        Constructor
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        lr_scheduler : None or Scheduler class object
            learning rate scheduler, defaults to None
        '''
        super(SGD_Optimizer, self).__init__(learning_rate)
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, Scheduler):
                raise Exception("lr_scheduler is required to be an object from "
                                +"Scheduler class")
        self.lr_scheduler = lr_scheduler
        
    def get_updates(self, cost, params):
        '''
        get gradient upates for the parameters
        
        Parameters
        ----------
        cost : symbolic expression
            cost of the objective, cost is a symbolic expression and needs to 
            be a scalar
        params : list of Parameter objects
            parameters of the model
        '''
        updates = OrderedDict()
        for param in params:
            tunable_param = param.get_tunable_parameters()
            for p in tunable_param:
                # Gradient update
                # 1. get lr coefficient for that parameter
                # 2. get parameter expression
                # 3. get gradient
                # 4. do gradient descent
                # 5. apply parameter specific constraint
                # 6. add updated parameter to parameter update
                
                lr_coeff = p.get_learning_rate_coefficient()
                crt_param = p.get_parameter_expression()
                crt_param_grad = LW.grad(cost, crt_param)
                updated_param = crt_param - self.lr * lr_coeff * crt_param_grad
                updated_param = param.apply_param_constraint(p.get_parameter_name(),
                                                             updated_param)
                updates[crt_param] = updated_param
        
        # 7. add learning rate update according to the scheduler
        if self.lr_scheduler is not None:
            updates.update(self.lr_scheduler.get_updates(self.lr))
            self.lr = self.lr_scheduler.rate
            
        return updates
# End SGD_Optimizer
################################################################################

################################################################################
# Begin SGD_Momentum_Optimizer
    
class SGD_Momentum_Optimizer(Optimizer):
    '''
    Stochastic gradient descent with momentum, subclass of Optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    mu : float [0, 1]
        momentum coefficient
    lr_scheduler : None or Scheduler class object
        learning rate scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    mu_scheduler : None or Scheduler class object
        momentum scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    '''
    def __init__(self, learning_rate, mu, lr_scheduler=None, mu_scheduler=None):
        '''
        Constructor
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        mu : float [0, 1]
            momentum coefficient
        lr_scheduler : None or Scheduler class object
            learning rate scheduler, defaults to None
        mu_scheduler : None or Scheduler class object
            momentum rate scheduler, defaults to None
        '''
        super(SGD_Momentum_Optimizer,self).__init__(learning_rate)
        self.mu = mu
        
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, Scheduler):
                raise Exception("lr_scheduler is required to be an object from "
                                +"Scheduler class")
        self.lr_scheduler = lr_scheduler
        
        if mu_scheduler is not None:
            if not isinstance(mu_scheduler, Scheduler):
                raise Exception("mu_scheduler is required to be an object from "
                                +"Scheduler class")
        self.mu_scheduler = mu_scheduler
        
    def get_updates(self, cost, params):
        '''
        get gradient upates for the parameters
        
        Parameters
        ----------
        cost : symbolic expression
            cost of the objective, cost is a symbolic expression and needs to 
            be a scalar
        params : list of Parameter objects
            parameters of the model
        '''
        updates = OrderedDict()
        for param in params:
            tunable_param = param.get_tunable_parameters()
            for p in tunable_param:
                # Gradient update
                # 1. add learning rate and momentum coefficient for the parameter
                # 2. get parameter value
                # 3. get gradient
                # 4. calculate momentum
                # 5. do gradient descent based on momentum
                # 6. apply parameter specific constraint
                # 7. add updated parameter to parameter update
                
                lr_coeff = p.get_learning_rate_coefficient()
                mu_coeff = p.get_momentum_coefficient()
                
                val = numpy.zeros(LW.shape(p.get_parameter_value()), dtype='float32')
                momentum = LW.data_variable(val)
                                    
                crt_param = p.get_parameter_expression()
                crt_param_grad = LW.grad(cost, crt_param)
                
                delta = self.mu*mu_coeff*momentum - self.lr*lr_coeff*crt_param_grad
                updated_param = crt_param + delta
                updated_param = param.apply_param_constraint(p.get_parameter_name(), 
                                                             updated_param)
                updates[crt_param] = updated_param
                updates[momentum] = delta
        
        # 8. add learning rate and momentum update according to the scheduler
        if self.lr_scheduler is not None:
            updates.update(self.lr_scheduler.get_updates(self.lr))
            self.lr = self.lr_scheduler.rate
        if self.mu_scheduler is not None:
            updates.update(self.mu_scheduler.get_updates(self.mu))
            self.mu = self.mu_scheduler.rate
            
        return updates
    
# End SGD_Momentum_Optimizer
################################################################################

################################################################################
# Begin SGD_Nesterov_Optimizer

class SGD_Nesterov_Optimizer(Optimizer):
    '''
    Stochastic gradient descent with Nesterov momentum, subclass of Optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    mu : float [0, 1]
        momentum coefficient
    lr_scheduler : None or Scheduler class object
        learning rate scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    mu_scheduler : None or Scheduler class object
        momentum scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    '''
    def __init__(self, learning_rate, mu, lr_scheduler=None, mu_scheduler=None):
        '''
        Constructor
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        mu : float [0, 1]
            momentum coefficient
        lr_scheduler : None or Scheduler class object
            learning rate scheduler, defaults to None
        mu_scheduler : None or Scheduler class object
            momentum rate scheduler, defaults to None
        '''
        super(SGD_Nesterov_Optimizer, self).__init__(learning_rate)
        self.mu = mu
        
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, Scheduler):
                raise Exception("lr_scheduler is required to be an object from "
                                +"Scheduler class")
        self.lr_scheduler = lr_scheduler
        
        if mu_scheduler is not None:
            if not isinstance(mu_scheduler, Scheduler):
                raise Exception("mu_scheduler is required to be an object from "
                                +"Scheduler class")
        self.mu_scheduler = mu_scheduler
        
    def get_updates(self, cost, params):
        '''
        get gradient upates for the parameters
        
        Parameters
        ----------
        cost : symbolic expression
            cost of the objective, cost is a symbolic expression and needs to 
            be a scalar
        params : list of Parameter objects
            parameters of the model
        '''
        updates = OrderedDict()
        for param in params:
            tunable_param = param.get_tunable_parameters()
            for p in tunable_param:
                # Gradient update
                # 1. get lr and mu coefficient for that parameter
                # 2. get parameter value
                # 3. get gradient
                # 4. calculate momentum
                # 5. do gradient descent based on momentum
                # 6. apply parameter specific constraint
                # 7. add updated parameter to parameter update
                
                lr_coef = p.get_learning_rate_coefficient()
                mu_coef = p.get_momentum_coefficient()
                
                val = numpy.zeros(LW.shape(p.get_parameter_value()), dtype='float32')
                momentum = LW.data_variable(val)
                    
                crt_param = param.get_parameter_expression()
                crt_param_grad = LW.grad(cost, crt_param)
                
                updated_vel = self.mu*mu_coef*momentum - self.lr*lr_coef*crt_param_grad
                inc = self.mu*mu_coef*updated_vel - self.lr*lr_coef*crt_param_grad
                
                updated_param = crt_param + inc
                updated_param = param.apply_param_constraint(p.get_parameter_name(), 
                                                            updated_param)
                updates[crt_param] = updated_param
                updates[momentum] = updated_vel
                
        # 8. add learning rate and momentum update according to the scheduler
        if self.lr_scheduler is not None:
            updates.update(self.lr_scheduler.get_updates(self.lr))
            self.lr = self.lr_scheduler.rate
        if self.mu_scheduler is not None:
            updates.update(self.mu_scheduler.get_updates(self.mu))
            self.mu = self.mu_scheduler.rate
        return updates
    
# End SGD_Nesterov_Optimizer
################################################################################

################################################################################
# Begin SGD_Rms_Optimizer

class SGD_Rms_Optimizer(Optimizer):
    '''
    Stochastic gradient descent using RMSprop 
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    
    subclass of Optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    lr_scheduler : None or Scheduler class object
        learning rate scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    gamma : float [0,1]
        exponential decay factor for the running RMS estimator of parameters
    '''
    def __init__(self, learning_rate, lr_scheduler=None, decay=0.9):
        '''
        Constructor
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        lr_scheduler : None or Scheduler class object
            learning rate scheduler, defaults to None
        decay : float [0,1]
            exponential decay factor for the running RMS estimator of parameters
            defaults to 0.9
        '''
        super(SGD_Rms_Optimizer, self).__init__(learning_rate)
        self.lr_scheduler = lr_scheduler
        self.gamma = decay

    def get_updates(self, cost, params):
        '''
        get gradient upates for the parameters
        
        Parameters
        ----------
        cost : symbolic expression
            cost of the objective, cost is a symbolic expression and needs to 
            be a scalar
        params : list of Parameter objects
            parameters of the model
        '''
        updates = OrderedDict()
        for param in params:
            tunable_param = param.get_tunable_parameters()
            for p in tunable_param:
                # Gradient update
                # 1. get lr coefficient for that parameter
                # 2. get parameter value
                # 3. get gradient
                # 4. calculate r value
                # 5. do gradient descent based on r values
                # 6. apply parameter specific constraint
                # 7. add updated parameter to parameter update
                
                lr_coef = p.get_learning_rate_coefficient()
                val = numpy.zeros(LW.shape(p.get_parameter_value()), dtype='float32')
                r = LW.data_variable(val)
                    
                crt_param = param.get_parameter_expression()
                crt_param_grad = LW.grad(cost, crt_param)
                
                updated_r = (1-self.gamma)*LW.square(crt_param_grad) + self.gamma*r + 1e-6 
                inc = - self.lr*lr_coef*crt_param_grad/LW.sqrt(updated_r)
                
                updated_param = crt_param + inc
                updated_param = param.apply_param_constraint(p.get_parameter_name(), 
                                                            updated_param)
                updates[crt_param] = updated_param
                updates[r] = updated_r
                
        
        # 8. add learning rate update according to the scheduler
        if self.lr_scheduler is not None:
            updates.update(self.lr_scheduler.get_updates(self.lr))
            self.lr = self.lr_scheduler.rate
        return updates
    
# End SGD_Rms_Optimizer
################################################################################

################################################################################
# Begin SGD_Adam_Optimizer
class SGD_Adam_Optimizer(Optimizer):
    '''
    Stochastic gradient descent using Adam by D. P. Kingma and J. L. Ba
    paper title "Adam: A method for stochastic optimization", ICLR 2015
    link: http://arxiv.org/pdf/1412.6980.pdf
    
    subclass of Optimizer
    
    Attributes
    ----------
    lr : float > 0
        learning rate
    lr_scheduler : None or Scheduler class object
        learning rate scheduler, adjust learning rate according to a predefined
        schedule, the behavior of the schedule depend on the provided 
        scheduler
    beta1 : float [0,1]
        exponential decay factor for the first momentum estimator
    beta2 : float [0,1]
        exponential decay factor for the second momentum estimator
    epsilon : small float
        some small float value to improve numerical stability of the algorithm
    '''
    def __init__(self, learning_rate, lr_scheduler=None,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        Constructor
        
        Parameters
        ----------
        learning_rate : float > 0
            learning rate
        lr_scheduler : None or Scheduler class object
            learning rate scheduler, defaults to None
        beta1 : float [0,1]
            exponential decay factor for the first momentum estimator
            defaults to 0.9
        beta2 : float [0,1]
            exponential decay factor for the first momentum estimator
            defaults to 0.9999
        epsilon : small float
            some small floating value to improve numerical stability of the 
            algorithm, defaults to 1e-8
        '''
        super(SGD_Adam_Optimizer, self).__init__(learning_rate)
        self.lr_scheduler = lr_scheduler
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, cost, params):
        '''
        get gradient upates for the parameters
        
        Parameters
        ----------
        cost : symbolic expression
            cost of the objective, cost is a symbolic expression and needs to 
            be a scalar
        params : list of Parameter objects
            parameters of the model
        '''
        t = LW.data_variable(numpy.asarray(0, dtype='float32'))
        updated_t = t + 1
        
        updates = OrderedDict()
        for param in params:
            tunable_param = param.get_tunable_parameters()
            for p in tunable_param:
                lr_coef = p.get_learning_rate_coefficient()
                
                crt_param = param.get_parameter_expression()
                crt_param_grad = LW.grad(cost, crt_param)
                
                param_shape = LW.shape(p.get_parameter_value())
                m = LW.data_variable(numpy.zeros(param_shape, dtype='float32'))
                v = LW.data_variable(numpy.zeros(param_shape, dtype='float32'))
                
                updated_m = self.beta1 * m + (1-self.beta1) * crt_param_grad
                updated_v = self.beta2 * v + (1-self.beta2) * LW.square(crt_param_grad)
                
                updated_lr = self.lr*lr_coef*LW.sqrt(1-LW.pow(self.beta2, updated_t))/(1-LW.pow(self.beta1, updated_t))
                
                
                inc = - updated_lr*updated_m/(LW.sqrt(updated_v) + self.epsilon)
                
                updated_param = crt_param + inc
                updated_param = param.apply_param_constraint(p.get_parameter_name(), 
                                                            updated_param)
                updates[crt_param] = updated_param
                updates[m] = updated_m
                updates[v] = updated_v
        
        updates[t] = updated_t
        if self.lr_scheduler is not None:
            updates.update(self.lr_scheduler.get_updates(self.lr))
            self.lr = self.lr_scheduler.rate
        return updates
    
# End SGD_Adam_Optimizer
################################################################################