import numpy
import theano
import copy
import time
import os
import abc

import theano.tensor as T

from model import Model
from utils.model.model_utils import raiseNotDefined
from data import UnlabeledDataProvider

__all__ = ['UnsupervisedModel']
class UnsupervisedModel(Model):
    
    @abc.abstractmethod
    def extract_feature_from_memory_data(self, data, *args, **kwargs):
        '''extract feature from data that already loaded into memory'''
        raiseNotDefined()
    
    # this method does not require subclass to implement
    def extract_feature_from_data_provider(self, data_provider, *args, **kwargs):
        '''extract feature using data provider'''
        raiseNotDefined()
    
    def validate(self, data_provider, n_valid_batches):
        valid_error = numpy.zeros((n_valid_batches,))
        for minibatch_index in xrange(n_valid_batches):
            (self.shared_valid, batch_start_idx, batch_end_idx) = \
                data_provider.get_valid_data_and_idx(minibatch_index)
            valid_error[minibatch_index] = self.validate_model(batch_start_idx, batch_end_idx)
        return numpy.mean(valid_error)
    
    def fit(self, data_provider, train_epoch, optimizer,
            valid_freq=None, noiseless_validation=True,
            dump_freq=-1, train_mean=None, batch_mean_subtraction=False,
            verbose=False, *args, **kwargs):
        
        '''fit model to data'''
        assert isinstance(data_provider, UnlabeledDataProvider), (
               'data_provider need to be a subclass from UnlabeledDataProvider'
               ' so that it provides appropriate data for unsupervised models')
        
        self.data_provider = data_provider
        self.nvalid_batches = data_provider.get_number_of_valid_batches()
        self.shared_train, _, _ = data_provider.get_train_data_and_idx(0)
        
        if self.nvalid_batches > 0:
            self.shared_valid, _, _ = data_provider.get_valid_data_and_idx(0)
            
        assert self.shared_train.dtype in ['uint8', 'float32'], (
               'shared_train should only have type uint8 or float32')
        self.uint8_data = (self.shared_train.dtype=='uint8')
        
        if train_mean is not None:
            self.train_mean = theano.shared(numpy.asarray(train_mean, dtype='float32'))
        else:
            self.train_mean = None
            
        self.batch_mean_subtraction = batch_mean_subtraction
        
        self.compile_functions(opt=optimizer, 
                               noiseless_validation=noiseless_validation,
                               *args, **kwargs)
    
        best_train_error = numpy.inf
        best_valid_error = numpy.inf
        epoch = 0
        best_iter = 0
        
        # cache for best parameters
        self.best_param_values = {}
        for l in self.layers:
            self.best_param_values[l.layerName] = {}
            layer_param_names = l.params.getAllParameterNames()
            for pn in layer_param_names:
                if not isinstance(l.params.getParameter(pn), T.TensorVariable):
                    self.best_param_values[l.layerName][pn] = copy.deepcopy(l.params.getParameterValue(pn))

        n_train_batches = data_provider.get_number_of_train_batches()
        n_valid_batches = data_provider.get_number_of_valid_batches()
        train_cost = numpy.zeros(n_train_batches)
        print n_train_batches

        if valid_freq is not None:
            assert valid_freq > 0, 'valid_freq need to be an integer greater than 0'
            valid_freq = valid_freq
        else:
            valid_freq = n_train_batches
            
        print '... training'
        start_time = time.clock()
        
        while (epoch < train_epoch):
            epoch_start = time.clock()
            
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                (self.shared_train, batch_start_idx, batch_end_idx) = \
                    data_provider.get_train_data_and_idx(minibatch_index)

                train_cost[minibatch_index] = self.train_model(batch_start_idx, batch_end_idx)
                                
                it = (epoch - 1) * n_train_batches + minibatch_index
                
                if self.nvalid_batches > 0 and (it+1) % valid_freq == 0:                              
                    this_valid_error = self.validate(data_provider, n_valid_batches)
                    print ('Epoch %d, Iteration %d, Validation error: %f' %(epoch, it + 1, this_valid_error)),
                    
                    if this_valid_error < best_valid_error:
                        best_valid_error = this_valid_error
                        self.copyParamsToBest()
                        best_iter = it
                        print '***',
                    print ''
                    
                if it != 0 and dump_freq>0 and it % dump_freq==0:
                    self.dump()

            this_train_error = numpy.mean(train_cost) 
            print ('Epoch %d, Training cost: %f' % (epoch, this_train_error)),
            
            if this_train_error < best_train_error:
                best_train_error = this_train_error
                if self.nvalid_batches <= 0: # in case of no validation
                    self.copyParamsToBest()
                print '<<<',

            print ', time cost: %.4f' % (time.clock() - epoch_start)
            
            
        end_time = time.clock()
        self.dump()
        print('Optimization complete.')
        print('Best valid score of %f  obtained at iteration %i' %
          (best_valid_error, best_iter + 1))
        self.copyBestToParams()
        print ('The code for file ' +
              os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        
        return best_valid_error
