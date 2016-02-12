import hickle
import abc
import copy

import theano.tensor as T

from utils.model.model_utils import raiseNotDefined

class Model:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, *args, **kwargs):
        self.initilize(*args, **kwargs)
        self.compile_model(*args, **kwargs)
        
    @abc.abstractmethod
    def compile_model(self, *args, **kwargs):
        '''functions may be used during both training and testing'''
        raiseNotDefined()
    
    @abc.abstractmethod
    def compile_functions(self, *args, **kwargs):
        '''functions used only during training'''
        raiseNotDefined()
    
    @abc.abstractmethod
    def initilize(self, *args, **kwargs):
        '''initialization of the model'''
        raiseNotDefined()
        
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        '''fit model to data'''
        raiseNotDefined()
        
    def save(self, filename):
        self.dump(filename)
        
    def load(self, filename):
        self.best_param_values = hickle.load(filename)
        self.copyBestToParams()
        
    def copyBestToParams(self):
        for l in self.layers:
            layer_param_names = l.params.getAllParameterNames()
            for pn in layer_param_names:
                if pn in self.best_param_values[l.layerName]:
                    l.params.setParameterValue(pn, self.best_param_values[l.layerName][pn])
            
    def copyParamsToBest(self):
        for l in self.layers:
            layer_param_names = l.params.getAllParameterNames()
            for pn in layer_param_names:
                if not isinstance(l.params.getParameter(pn), T.TensorVariable):
                    self.best_param_values[l.layerName][pn] = copy.deepcopy(l.params.getParameterValue(pn))
            
    def dump(self, filename='tmp_model.joblib'):
        # Saves params as numpy values since saving as theano variables was
        # found to display unstable beahavior during the save.
        # re-conversion to a theano variable is handled in obtain_network().
        hickle.dump(self.best_param_values, filename, mode='w', compression='gzip')
