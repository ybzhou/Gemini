import hickle
import abc
import copy

import theano.tensor as T
import layers

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
        
    def network_fprop(self, network_layers, x, y=None, isTest = False, noiseless=False):
        layer_outputs = {}
        if isTest:
            mode = 'test'
        else:
            mode = 'train'
            
        for layer_idx in xrange(len(network_layers)):
            crt_layer = network_layers[layer_idx]
            
            if isinstance(crt_layer, layers.DataLayer):
                if crt_layer.inputType == 'data':
                    layer_outputs[crt_layer.layerName] = crt_layer.fprop(x)
                elif crt_layer.inputType == 'label':
                    layer_outputs[crt_layer.layerName] = crt_layer.fprop(y)
                else:
                    raise('unkown layer input type')
            else:
                
                if noiseless and isinstance(crt_layer, layers.NoiseLayer):
                    prev_noise_level = crt_layer.noiseLevel
                    crt_layer.noiseLevel = 0
                    
                prev_layers = crt_layer.getPreviousLayer()
            
                # only concatenate layers takes multiple inputs
                if len(prev_layers) > 1:
                    input_for_crt_layer = []
                    for l in prev_layers:
                        input_for_crt_layer.append(layer_outputs[l.layerName])
                else:
                    input_for_crt_layer = layer_outputs[prev_layers[0].layerName]
                
                if isinstance(crt_layer, layers.BatchNormLayer):
                    output_for_crt_layer = crt_layer.fprop(input_for_crt_layer, mode)
                elif isinstance(crt_layer, layers.DropoutLayer) \
                   or isinstance(crt_layer, layers.BatchStandardizeLayer):
                    output_for_crt_layer = crt_layer.fprop(input_for_crt_layer, isTest)
                else:
                    output_for_crt_layer = crt_layer.fprop(input_for_crt_layer)
                
                layer_outputs[crt_layer.layerName] = output_for_crt_layer
                
                if noiseless and isinstance(crt_layer, layers.NoiseLayer):
                    crt_layer.noiseLevel = prev_noise_level
                    
        return layer_outputs
