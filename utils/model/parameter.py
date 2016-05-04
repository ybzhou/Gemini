import copy
import numpy
import libwrapper as LW

from .regularizers import Regularizer
from .constraints import NormConstraint

class Parameter(object):
    '''
    Class that represent parameter
    
    Attributes
    ----------
    param_name : str
        the name of the parameter
    param : libwrapper.data_variable_type or libwrapper.symbolic_variable_type
        symbolic expression for the parameter
    regularizer : utils.model.Regularizer or None
        regularizer for the current parameter
    constaint : utils.model.NormConstraint or None
        constraint for the current parameter
    lr_coefficient : float
        coefficient of the parameter that is going to get multiplied to the 
        specified global learning rate
    can_tune : boolean
        specify if the parameter is tunable
    can_copy : boolean
        specify is the parameter is copiable
    '''
    def __init__(self, param_name, param_expr, regularizer=None, constraint=None, 
                 lr_coefficient=1.0, mu_coefficient=1.0,
                 can_tune=True, can_copy=True):
        '''
        Constructor
        
        Parameters
        ----------
        param_name : str
            the name of the parameter
        param : libwrapper.data_variable_type or libwrapper.symbolic_variable_type
            symbolic expression for the parameter
        regularizer : utils.model.Regularizer or None
            regularizer for the current parameter, defaults to no regularizer,
            i.e. None
        constaint : utils.model.NormConstraint or None
            constraint for the current parameter, defaults to no constraint, 
            i.e. None
        lr_coefficient : float > 0
            coefficient of the parameter that is going to get multiplied to the 
            specified global learning rate, defaults to 1.0
        mu_coefficient : float > 0
            coefficient of the parameter that is going to get multiplied to the 
            specified global momentum, defaults to 1.0
        can_tune : boolean
            specify if the parameter is tunable, defaults to True
        can_copy : boolean
            specify is the parameter is copiable, defaults to True
        '''
        if (not isinstance(param_expr, LW.data_variable_type)
            and not isinstance(param_expr, LW.symblic_variable_type)):
            raise Exception("param_expr is required to be either of "
                            +"libwrapper.data_variable_type or "
                            +"libwrapper.symbolic_variable_type")
            
        if not isinstance(regularizer, Regularizer):
            raise Exception("regularizer is required to be an object of class "
                            +"Regularizer")
            
        if not isinstance(constraint, NormConstraint):
            raise Exception("constraint is required to be an object of class "
                            +"NormConstraint")
            
        self.param_name = param_name
        self.param = param_expr
        self.reg = regularizer
        self.cons = constraint
        self.lr_coef = lr_coefficient
        self.mu_coef = mu_coefficient
        self.can_tune = can_tune
        self.can_copy = can_copy
    
    def get_parameter_name(self):
        return self.param_name
    
    def get_parameter_expression(self):
        return self.param
    
    def get_parameter_value(self):
        if self.can_copy:
            return numpy.asarray(LW.get_value(self.param))
        else:
            raise Exception("Only copiable parameter contains value")
    
    def set_parameter_value(self, value):
        if self.can_copy:
            LW.set_value(self.param, value)
        else:
            raise Exception("Can only set value for copiable parameter")
    
    def get_regularizer(self):
        '''
        getter for regularizer
        '''
        return self.reg
    
    def set_regularizer(self, regularizer):
        '''
        setter for regularizer
        '''
        if not isinstance(regularizer, Regularizer):
            raise Exception("regularizer is required to be an object of class "
                            +"Regularizer")
            
        self.reg = regularizer
        
    def get_constraint(self):
        '''
        getter for constraint
        '''
        return self.cons
    
    def set_constraint(self, constraint):
        '''
        setter for constraint
        '''
        if not isinstance(constraint, NormConstraint):
            raise Exception("constraint is required to be an object of class "
                            +"NormConstraint")
        self.cons = constraint
    
    def get_learning_rate_coefficient(self):
        '''
        getter for learning rate coefficient
        '''
        return self.lr_coef
    
    def set_learing_rate_coefficient(self, lr_coef):
        '''
        setter for learning rate coefficient
        '''
        self.lr_coef = lr_coef
        
    def get_momentum_coefficient(self):
        '''
        getter for learning rate coefficient
        '''
        return self.mu_coef
    
    def set_momentum_coefficient(self, mu_coef):
        '''
        setter for learning rate coefficient
        '''
        self.mu_coef = mu_coef
        
    def get_is_tunable(self):
        '''
        getter for tunable
        '''
        return self.can_tune
    
    def set_is_tunable(self, can_tune):
        '''
        setter for tunable
        '''
        self.can_tune = can_tune
    
    def get_is_copiable(self):
        '''
        getter for copiable
        '''
        return self.can_copy
        
    def set_is_copiable(self, can_copy):
        '''
        setter for copiable
        '''
        self.can_copy = can_copy
        
        
class LayerParameter(object):
    '''
    Class that represent parameters for a layer
    
    Attributes
    ----------
    params : a dictionary of Parameter objects
        dictionary of Parameter objects that represent all parameters for the 
        layer, the key is the name of the parameter and the value is the 
        corresponding parameter class object
    '''
    def __init__(self, params = []):
        """
        Constructor
        
        Parameters
        ----------
        params : list of Parameter objects
            list of Parameter objects that represent all parameters for the layer, 
            defaults to an empty list
        """
        
        if type(params) is not list:
            raise Exception("params need to be a list of Parameter object")
        
        if params:
            for p in params:
                if not isinstance(p, Parameter):
                    raise Exception("object within params list need to be object of Parameter class")
        
        self.params = {}
        for p in params:
            pname = p.get_parameter_name()
            if pname in self.params:
                raise Exception("Cannot have parameter of the same name in the "
                                +"same layer, duplicate name: "+pname)
            self.params[pname] = p
    
    def add_parameter(self, parameter):
        if not isinstance(parameter, Parameter):
            raise Exception("parameter need to be an object of Parameter class")
        pname = parameter.get_parameter_name()
        if pname in self.params:
            raise Exception("Cannot have parameter of the same name in the "
                            +"same layer, duplicate name: "+pname)
        self.params[pname] = parameter
    
    def get_all_parameters(self):
        return self.params.values()
    
    def get_parameter(self, pname):
        return self.params[pname]
    
    def get_parameter_value(self, pname):
        return self.params[pname].get_parameter_value()

    def get_num_tunable_params(self):
        cnt = 0
        for p in self.params.keys():
            para = self.params[p]
            if para.get_is_tunable():
                cnt += 1
        return cnt
    
    def get_total_num_params(self):
        return len(self.params.keys())
            
    def get_tunable_parameters(self):
        param = []
        for pn in self.params.keys():
            if self.params[pn].get_is_tunable():
                param.append(self.params[pn])
        return param
    
    def get_param_regularization(self):
        reg = 0.
        for pn in self.params:
            if self.params[pn].get_regularizer() is not None:
                reg += self.params[pn].get_regularizer().get_regularization(
                            self.params[pn].get_parameter_expression())
        return reg
    
    def apply_param_constraint(self, pname, value):
        p = self.params[pname]
        updated_value = value
        if p.get_constraint() is not None:
            updated_value = p.get_constraint().apply_constraint(value)
        return updated_value
    
    