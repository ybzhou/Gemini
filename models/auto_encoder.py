import theano
import numpy

import theano.tensor as T

from utils.model.model_utils import obtain_network, validate_network
from unsupervised_model import UnsupervisedModel
from math import ceil
from data import UnlabeledDataProvider

__all__ = ['AutoEncoder']

class AutoEncoder(UnsupervisedModel):
    
    """ takes network structure and model parameters and assigns class attributes """
    def initilize(self, network_structure, batch_size,
                  seed, network_cost, init_params = {}, *args, **kwargs):
        self.ae_structure = network_structure
        self.batch_size = batch_size
        self.seed = seed
        self.network_cost = network_cost
        
        self.init_params = init_params
    
    """ network structure definition -> network stucture + post training functions """
    def compile_model(self, *args, **kwargs):
        print '... building the model'
        self.ae_name_index_dic = validate_network(self.ae_structure)
        
        self.x = T.matrix('x')
        
        self.layers = obtain_network(self.batch_size,
                                     self.ae_structure, 
                                     self.ae_name_index_dic, self.init_params)

        self.params = [l.params for l in self.layers]
        
    """ training specific functions - rename function - TODO"""
    def compile_functions(self, opt, noiseless_validation=True, **args):
        print '... compiling training functions'
        
        # propagte for training with batch normalization with upated std and mean for each batch
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=False, noiseless=False)
        cost, show_cost = self.get_cost(layer_outputs, self.layers)
        self.opt = opt
        updates = self.opt.get_updates(cost, self.params)
        
        start_index, end_index = T.iscalars('s_i', 'e_i')
        if self.uint8_data:
            given_train_x = T.cast(self.shared_train[start_index:end_index], dtype='float32')
        else:
            given_train_x = self.shared_train[start_index:end_index]
            
        if self.batch_mean_subtraction:
            assert self.train_mean is not None, 'train_mean cannot be None for batch mean subtraction'
            given_train_x -= self.train_mean
            
        self.train_model = theano.function( inputs=[start_index, end_index], 
                                            outputs=show_cost, updates = updates,
                                            givens = {
                                                      self.x: given_train_x,
                                                      }
                                           )
        
        
        if self.nvalid_batches>0:
            layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseLess=noiseless_validation)
            final_output = layer_outputs[self.ae_structure[-1]['name']]
            _, show_cost = self.get_cost(layer_outputs, self.layers)
    
            if self.uint8_data:
                given_valid_x = T.cast(self.shared_valid[start_index:end_index], dtype='float32')
            else:
                given_valid_x = self.shared_valid[start_index:end_index]
                
            if self.batch_mean_subtraction:
                assert self.train_mean is not None, 'train_mean cannot be None for batch mean subtraction'
                given_valid_x -= self.train_mean
                
            self.validate_model = theano.function(inputs=[start_index, end_index], 
                                               outputs=show_cost,
                                                givens = {
                                                          self.x: given_valid_x,
                                                          }
                                              )
    
    def get_cost(self, layer_outputs, layers):
        # get network cost
        cost = 0.
        for layer_name in self.network_cost.keys():
            target_name, cost_func = self.network_cost[layer_name]
            cost += cost_func.getCost(layer_outputs[target_name], 
                                      layer_outputs[layer_name])
        # get parameter/regularizer cost
        reg = 0.
        for layer in layers:
            reg += layer.getParamRegularization()
        
        return cost + reg, cost
            
        
    def reconstruct(self, data, noiseLess=False):
        
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseless=noiseLess)
         
        # the reconstruction is always the last layer of the network
        final_output = layer_outputs[self.ae_structure[-1]['name']]
         
        self.__reconstruct = theano.function([self.x], final_output)
        
        ndata = data.shape[0]
        nbatches = int(ceil(ndata/self.batch_size))
        recs = numpy.zeros(data.shape, dtype='float32')
        
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > ndata:
                batch_end = ndata
            
            recs[batch_start:batch_end] = self.__reconstruct(data[batch_start:batch_end]).reshape((batch_end-batch_start, data.shape[1]))
                    
        return recs
    
    def extract_feature_from_memory_data(self, data, feature_layer_name, niter=1, noiseless=False):
        assert len(data.shape) == 2, ('data should be passed in as a matrix, '
                                      'where each row represent one example')
        
        assert feature_layer_name in self.ae_name_index_dic, ('need to provide feature_layer_name '
                                                           'that is in the current network structure')
        ndata = data.shape[0]
        trim_pred = False
        if ndata < self.batch_size:
            trim_pred = True
            true_ndata = ndata
            # pad the rest with zeros
            data = numpy.vstack((data, numpy.zeros((self.batch_size-ndata, data.shape[1]), dtype=data.dtype)))
            ndata = self.batch_size
        
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseless=noiseless)
        h_given_x = layer_outputs[feature_layer_name]

        extract_feature = theano.function([self.x], h_given_x)
        
        
        nbatches = int(ceil(float(ndata)/self.batch_size))
        features = numpy.zeros((ndata,)+self.layers[self.ae_name_index_dic[feature_layer_name]].getOutputShape()[1:], 
                                 dtype='float32')
        
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > ndata:
                batch_end = ndata
                batch_start = batch_end-self.batch_size
            
            crt_data = data[batch_start:batch_end]
            
            for j in xrange(niter):
                if j == 0:
                    p = extract_feature(crt_data)
                else:
                    p += extract_feature(crt_data)
            
            features[batch_start:batch_end] = p/float(niter)
        
        
        if trim_pred:
            features = features[:true_ndata]
            
        return features
    
    
    def extract_feature_from_data_provider(self, data_provider, feature_layer_name, 
                                           train_mean=None, batch_mean_subtraction=False, 
                                           niter=1, noiseless=False):
        assert isinstance(data_provider, UnlabeledDataProvider), (
               'data_provider need to be a subclass from UnlabeledDataProvider'
               ' so that it provides appropriate data for unsupervised models')
        
        assert feature_layer_name in self.ae_name_index_dic, ('need to provide feature_layer_name '
                                                           'that is in the current network structure')
        
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseless=noiseless)
        h_given_x = layer_outputs[feature_layer_name]

        
        self.shared_train, _, _ = data_provider.get_train_data_and_idx(0)
        start_index, end_index = T.iscalars('s_i', 'e_i')
        xgiven = self.shared_train[start_index:end_index]
        if self.shared_train.dtype=='uint8':
            xgiven = T.cast(xgiven, dtype='float32')
        
        if train_mean is not None and batch_mean_subtraction:
            tm = theano.shared(numpy.asarray(train_mean, dtype='float32'))
            xgiven -= tm
            
        extract_feature = theano.function([start_index, end_index], 
                                         h_given_x,
                                         givens={self.x:xgiven})
        
        ndata = data_provider.get_number_of_train_data()
        
        features = numpy.zeros((ndata,)+self.layers[self.ae_name_index_dic[feature_layer_name]].getOutputShape()[1:], 
                                 dtype='float32')
        
        for minibatch_idx in xrange(data_provider.get_number_of_train_batches()):
            self.shared_train, s_i, e_i = data_provider.get_train_data_and_idx(minibatch_idx)
            pred_start = minibatch_idx*self.batch_size
            pred_end = (minibatch_idx+1)*self.batch_size
            if pred_end > ndata:
                pred_start = ndata-self.batch_size
                pred_end = ndata
            
            for j in xrange(niter):
                if j == 0:
                    p = extract_feature(s_i, e_i)
                else:
                    p += extract_feature(s_i, e_i)
                    
            features[pred_start:pred_end] = p/float(niter)
            
        return features
    
    def reconstruct_from_memory_data(self, data, steps=1, noiseless=True):
        assert len(data.shape) == 2, ('data should be passed in as a matrix, '
                                      'where each row represent one example')
        assert ((steps==1) == noiseless), (
                'need noise to simulate generalized deonising autoencoder, '
                'for noiseless case there is no need to taking multiple steps '
                'in reconstruction')
        
        ndata = data.shape[0]
        trim_pred = False
        if ndata < self.batch_size:
            trim_pred = True
            true_ndata = ndata
            # pad the rest with zeros
            data = numpy.vstack((data, numpy.zeros((self.batch_size-ndata, data.shape[1]), dtype=data.dtype)))
            ndata = self.batch_size
        
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseless=noiseless)
         
        # the reconstruction is always the last layer of the network
        x_hat_given_x = layer_outputs[self.ae_structure[-1]['name']]
         
        reconstruct = theano.function([self.x], x_hat_given_x)
        
        
        nbatches = int(ceil(float(ndata)/self.batch_size))
        recs = numpy.zeros(data.shape, dtype='float32')
        
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > ndata:
                batch_end = ndata
                batch_start = batch_end-self.batch_size
            
            crt_data = data[batch_start:batch_end]
            
            for j in xrange(steps): # 
                if j == 0:
                    p = reconstruct(crt_data)
                else:
                    p = reconstruct(p)
                p = p.reshape((self.batch_size, -1))
            recs[batch_start:batch_end] = p.reshape((self.batch_size,-1))
        
        
        if trim_pred:
            recs = recs[:true_ndata]
            
        return recs
    
    def reconstruct_from_data_provider(self, data_provider, train_mean=None, 
                                       batch_mean_subtraction=False, steps=1, noiseless=True):
        assert isinstance(data_provider, UnlabeledDataProvider), (
               'data_provider need to be a subclass from UnlabeledDataProvider'
               ' so that it provides appropriate data for unsupervised models')
        
        assert ((steps==1) == noiseless), (
                'need noise to simulate generalized deonising autoencoder, '
                'for noiseless case there is no need to taking multiple steps '
                'in reconstruction')
        
        layer_outputs = self.network_fprop(self.layers, self.x, isTest=True, noiseless=noiseless)
         
        # the reconstruction is always the last layer of the network
        x_hat_given_x = layer_outputs[self.ae_structure[-1]['name']]

        self.shared_train, _, _ = data_provider.get_train_data_and_idx(0)
        start_index, end_index = T.iscalars('s_i', 'e_i')
        xgiven = self.shared_train[start_index:end_index]
        if self.shared_train.dtype=='uint8':
            xgiven = T.cast(xgiven, dtype='float32')
        
        if train_mean is not None and batch_mean_subtraction:
            tm = theano.shared(numpy.asarray(train_mean, dtype='float32'))
            xgiven -= tm
            
        reconstruct_dp = theano.function([start_index, end_index], 
                                         x_hat_given_x,
                                         givens={self.x:xgiven})
        
        reconstruct_mem = theano.function([self.x], x_hat_given_x)
        
        ndata = data_provider.get_number_of_train_data()
        
        recs = numpy.zeros((ndata, self.shared_train.get_value().shape[1]), 
                                 dtype='float32')
        
        for minibatch_idx in xrange(data_provider.get_number_of_train_batches()):
            self.shared_train, s_i, e_i = data_provider.get_train_data_and_idx(minibatch_idx)
            pred_start = minibatch_idx*self.batch_size
            pred_end = (minibatch_idx+1)*self.batch_size
            if pred_end > ndata:
                pred_start = ndata-self.batch_size
                pred_end = ndata
            
            for j in xrange(steps):
                if j == 0:
                    p = reconstruct_dp(s_i, e_i)
                else:
                    p = reconstruct_mem(p)
                    
            recs[pred_start:pred_end] = p
            
        return recs