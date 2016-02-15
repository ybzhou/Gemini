import theano
import layers
import numpy

import theano.tensor as T

from utils.model.model_utils import obtain_network, validate_network
from models.supervised_model import SupervisedModel
from math import ceil
from data import LabeledDataProvider

__all__ = ['MIMLP']

class MIMLP(SupervisedModel):
    
    """ takes network structure and model parameters and assigns class attributes """
    def initilize(self, network_structure, ninputs,
                  batch_size, seed, network_cost, init_params={}, *args, **kwargs):
        self.ninputs = ninputs
        self.network_structure = network_structure
        self.batch_size = batch_size
        self.seed = seed
        self.network_cost = network_cost        
        
        # find label layer and see if the label is vector valued
        for ls  in self.network_structure:
            if ls['layer_type'] == 'data' and ls['input_type'] == 'label':
                if 'input_shape' in ls:
                    self.label_dims = len(ls['input_shape'])
                else:
                    self.label_dims = 1
        
        assert hasattr(self, 'label_dims'), 'Need to specify label input for MLP'
        
        self.xs = []
        for i in xrange(self.ninputs):
            self.xs.append(T.matrix('x_%d' % i))
            
        if self.label_dims == 2:
            self.y = T.imatrix('y')
        elif self.label_dims == 1:
            self.y = T.ivector('y')
        elif self.label_dims == 4:
            self.y = T.tensor4('y')
        else:
            raise 'Currently only support scalar, vector or tensor-4 labels'

        self.init_params = init_params
    
    """ network structure definition -> network stucture + post training functions """
    def compile_model(self, *args, **kwargs):
        print '... building the model'
        self.name_index_dic = validate_network(self.network_structure)
        
        self.layers = obtain_network(self.batch_size,
                                     self.network_structure, 
                                     self.name_index_dic, self.init_params)

        self.params = [l.params for l in self.layers]
        
        
    """ training specific functions - rename function - TODO"""
    def compile_functions(self, opt, **args):
        print '... compiling training functions'
        
        # propagte for training with batch normalization with upated std and mean for each batch
        self.layer_outputs = self.network_fprop()
        cost, show_cost = self.get_cost()
        self.opt = opt
        updates = self.opt.get_updates(cost, self.params)
        
        # propagate again for validation with fixed mean and std for batch normalization
        self.layer_outputs = self.network_fprop(isTest=True, noiseless=True)
        self.final_output = self.layer_outputs[self.network_structure[-1]['name']]
        errors = self.get_errors()
        
        start_index = T.iscalar('start_index')
        end_index = T.iscalar('end_index')
        
        train_given = {}
        print 'number of training inputs = ', self.ninputs
        for i in xrange(self.ninputs):
            if self.uint8_data:
                train_given[self.xs[i]] = T.cast(self.shared_train[i][start_index:end_index], dtype='float32')
            else:
                train_given[self.xs[i]] = self.shared_train[i][start_index:end_index]
                
            if self.batch_mean_subtraction:
                assert self.train_mean is not None, 'train_mean cannot be None for batch mean subtraction'
                assert len(self.train_mean) == self.ninputs, 'train_mean need to have the same number as number of inputs'
                train_given[self.xs[i]] -= self.train_mean[i]
            
        train_given[self.y] = self.shared_train_labels[start_index:end_index]
        
        self.train_model = theano.function( inputs=[start_index, end_index], 
                                            outputs=[show_cost, errors], updates = updates,
                                            givens = train_given
                                           )
        
        if hasattr(self, 'shared_valid'):
            valid_given = {}
            for i in xrange(self.ninputs):
                if self.uint8_data:
                    valid_given[self.xs[i]] = T.cast(self.shared_valid[i][start_index:end_index], dtype='float32')
                else:
                    valid_given[self.xs[i]] = self.shared_valid[i][start_index:end_index]
                
                if self.batch_mean_subtraction:
                    assert self.train_mean is not None, 'train_mean cannot be None for batch mean subtraction'
                    assert len(self.train_mean) == self.ninputs, 'train_mean need to have the same number as number of inputs'
                    valid_given[self.xs[i]] -= self.train_mean[i]
                
            valid_given[self.y] = self.shared_valid_labels[start_index:end_index]
            
            self.validate_model = theano.function( inputs=[start_index,end_index], 
                                                   outputs=errors,
                                                    givens = valid_given
                                                  )
    
    def network_fprop(self, isTest = False, noiseless=False):
        layer_outputs = {}
        if isTest:
            mode = 'test'
        else:
            mode = 'train'
            
        for layer_idx in xrange(len(self.layers)):
            crt_layer = self.layers[layer_idx]
            
            if isinstance(crt_layer, layers.DataLayer):
                if crt_layer.inputType == 'data':
                    layer_outputs[crt_layer.layerName] = crt_layer.fprop(self.xs[crt_layer.data_index])
                elif crt_layer.inputType == 'label':
                    layer_outputs[crt_layer.layerName] = crt_layer.fprop(self.y)
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
    
    def get_cost(self):
        # get network cost
        cost = 0.
        for layer_name in self.network_cost.keys():
            target_name, cost_func = self.network_cost[layer_name]
            cost += cost_func.getCost(self.layer_outputs[target_name], 
                                      self.layer_outputs[layer_name])
        # get parameter/regularizer cost
        reg = 0.
        for layer in self.layers:
            reg += layer.getParamRegularization()
        
        return cost + reg, cost
        
    def get_errors(self):
        return self.custom_error.getError(self.y, self.final_output)
    
    def extract_feature_from_memory_data(self, data, feature_layer_name, niter=1, noiseless=False):
        assert len(data[0].shape) == 2, ('data should be passed in as a matrix, '
                                      'where each row represent one example')
        
        assert feature_layer_name in self.name_index_dic, ('need to provide feature_layer_name '
                                                           'that is in the current network structure')
        ndata = data[0].shape[0]
        trim_pred = False
        if ndata < self.batch_size:
            trim_pred = True
            true_ndata = ndata
            # pad the rest with zeros
            for i in xrange(self.ninputs):
                data[i] = numpy.vstack((data[i], numpy.zeros((self.batch_size-ndata, data.shape[1]), dtype=data.dtype)))
            ndata = self.batch_size
            
        layer_outputs = self.network_fprop(isTest=True, noiseless=noiseless)
        
        final_output = layer_outputs[feature_layer_name]

        self.__predict = theano.function(self.xs, final_output)
        
        
        nbatches = int(ceil(float(ndata)/self.batch_size))
        prediction = numpy.zeros((ndata,)+self.layers[self.name_index_dic[feature_layer_name]].getOutputShape()[1:], 
                                 dtype='float32')
        
        crt_data = [None]*self.ninputs
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > ndata:
                batch_end = ndata
                batch_start = batch_end-self.batch_size
            
            for j in xrange(self.ninputs):
                crt_data[j] = data[j][batch_start:batch_end]
            
            for j in xrange(niter):
                if j == 0:
                    p = self.__predict(*crt_data)
                else:
                    p += self.__predict(*crt_data)
            
            prediction[batch_start:batch_end] = p
        
        
        if trim_pred:
            prediction = prediction[:true_ndata]
            
        return prediction
    
    
    def extract_feature_from_data_provider(self, data_provider, feature_layer_name, 
                                           train_mean=None, batch_mean_subtraction=False, 
                                           niter=1, noiseless=False):
        assert isinstance(data_provider, LabeledDataProvider), (
               'data_provider need to be a subclass from LabeledDataProvider'
               ' so that it provides labeled data for supervised models')
        
        assert feature_layer_name in self.name_index_dic, ('need to provide feature_layer_name '
                                                           'that is in the current network structure')
        
        layer_outputs = self.network_fprop(isTest=True, noiseless=noiseless)
        
        # assumes the output is always the last layer of the network for now
        final_output = layer_outputs[feature_layer_name]
        
        self.shared_train, _, _, _ = data_provider.get_train_labeled_data_and_idx(0)
        start_index, end_index = T.iscalars('s_i', 'e_i')
        xgiven = {}
        for i in xrange(self.ninputs):
            xgiven[self.xs[i]] = self.shared_train[i][start_index:end_index]
            
        if self.shared_train[0].dtype=='uint8':
            for i in xrange(self.ninputs):
                xgiven[self.xs[i]] = T.cast(xgiven[self.xs[i]], dtype='float32')
        
        if train_mean is not None and batch_mean_subtraction:
            tm = [None] * self.ninputs
            for i in xrange(self.ninputs):
                tm[i] = theano.shared(numpy.asarray(train_mean[i], dtype='float32'))
                xgiven[self.xs[i]] -= tm[i]
            
        self.__predict = theano.function([start_index, end_index], 
                                         final_output,
                                         givens=xgiven)
        
        ndata = data_provider.get_number_of_train_data()
        
        prediction = numpy.zeros((ndata,)+self.layers[self.name_index_dic[feature_layer_name]].getOutputShape()[1:], 
                                 dtype='float32')
        
        for minibatch_idx in xrange(data_provider.get_number_of_train_batches()):
            self.shared_train, _, s_i, e_i = data_provider.get_train_labeled_data_and_idx(minibatch_idx)
            pred_start = minibatch_idx*self.batch_size
            pred_end = (minibatch_idx+1)*self.batch_size
            if pred_end > ndata:
                pred_start = ndata-self.batch_size
                pred_end = ndata
            
            for j in xrange(niter):
                if j == 0:
                    p = self.__predict(s_i, e_i)
                else:
                    p += self.__predict(s_i, e_i)
                    
            prediction[pred_start:pred_end] = p/float(niter)
            
        return prediction
        
    def predict_from_memory_data(self, data, pred_layer_name='', niter=1, noiseless=False):
        '''
           predict from the layer with name pred_layer_name, and it is assumed 
           that larger values are more probable
        '''
        if pred_layer_name == '':
            # default to last layer's name
            pred_layer_name = self.network_structure[-1]['name']
            
        pred = self.extract_feature_from_memory_data(data=data,
                                              feature_layer_name=pred_layer_name,
                                              niter=niter,
                                              noiseless=noiseless)
        
        return numpy.argmax(pred, axis=1)
    
    def predict_from_data_provider(self, data_provider, 
                                   train_mean=None, batch_mean_subtraction=False,
                                   pred_layer_name='', niter=1, noiseless=False):
        '''
           predict from the layer with name pred_layer_name, and it is assumed 
           that larger values are more probable
        '''
        
        if pred_layer_name == '':
            # default to last layer's name
            pred_layer_name = self.network_structure[-1]['name']
        
        pred = self.extract_feature_from_data_provider(data_provider=data_provider, 
                                                       train_mean=train_mean,
                                                       batch_mean_subtraction=batch_mean_subtraction,
                                                       feature_layer_name=pred_layer_name, 
                                                       niter=niter, 
                                                       noiseless=noiseless)
        return numpy.argmax(pred, axis=1)