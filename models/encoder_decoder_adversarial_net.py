import theano
import layers
import numpy

import time
import copy
import os

import theano.tensor as T
import theano.sandbox.rng_mrg as MRG

from utils.model.model_utils import obtain_network, validate_network
from auto_encoder import AutoEncoder
from math import ceil
from data import UnlabeledDataProvider
from scipy.misc import imsave

__all__ = ['EncDecAN']

class EncDecAN(AutoEncoder):
    """ takes network structure and model parameters and assigns class attributes """
    def initilize(self, encoder_ns, decoder_ns, data_discrim_ns, prior_discrim_ns,
                  noise_func, 
                  batch_size, num_z, seed, 
                  ae_cost_func,
                  init_params={},
                  *args, **kwargs):
        self.encoder_ns = encoder_ns
        self.decoder_ns = decoder_ns
        self.data_discrim_ns = data_discrim_ns
        self.prior_discrim_ns = prior_discrim_ns
        self.batch_size = batch_size
        self.seed = seed
        self.init_params = init_params
        self.theano_rng = MRG.MRG_RandomStreams(seed)
        self.num_z = num_z
        self.noise_func = noise_func
        self.cost_func = ae_cost_func
        
        
    def compile_model(self, *args, **kwargs):
        print '... building the model'
        data_dis_names = [d['name'] for d in self.data_discrim_ns]
        prior_dis_names = [d['name'] for d in self.prior_discrim_ns]
        enc_names = [d['name'] for d in self.encoder_ns]
        dec_names = [d['name'] for d in self.decoder_ns]
        assert len(set(prior_dis_names+data_dis_names+enc_names+dec_names)) == len(data_dis_names)+len(prior_dis_names) + len(enc_names) + len(dec_names), \
        'layer name cannot be the same for encoder, decoder and discriminator network structure'
        
        total_structure = []
        total_structure.extend(self.encoder_ns)
        total_structure.extend(self.decoder_ns)
        total_structure.extend(self.data_discrim_ns)
        total_structure.extend(self.prior_discrim_ns)
        self.name_index_dic = validate_network(total_structure)
        
        self.x = T.matrix('x')
        self.z = T.matrix('z')
        
        self.layers = obtain_network(self.batch_size,
                                     total_structure, 
                                     self.name_index_dic, self.init_params,
                                     check_output_usage=False)
        
        # encoder
        sep_index1 = self.name_index_dic[self.decoder_ns[0]['name']]
        self.enc_layers = self.layers[:sep_index1]
        # decoder
        sep_index2 = self.name_index_dic[self.data_discrim_ns[0]['name']]
        self.dec_layers = self.layers[sep_index1:sep_index2]
        # data discriminator
        sep_index3 = self.name_index_dic[self.prior_discrim_ns[0]['name']]
        self.data_dis_layers = self.layers[sep_index2:sep_index3]
        # prior discriminator
        self.prior_dis_layers = self.layers[sep_index3:]
        
        self.enc_params = [l.params for l in self.enc_layers]
        self.dec_params = [l.params for l in self.dec_layers]
        self.data_dis_params = [l.params for l in self.data_dis_layers]
        self.prior_dis_params = [l.params for l in self.prior_dis_layers]
        
        
    def compile_functions(self, opt, noiseless_validation=True, **args):
        print '... compiling training functions'
        
        (prior_gen_cost, prior_gen_show_cost, 
         prior_dis_cost, prior_dis_show_cost, 
         data_gen_cost, data_gen_show_cost,
         data_dis_cost, data_dis_show_cost,
         rec_cost, rec_show_cost) = self.get_cost(isTest=False)
           
        self.opt = opt
        prior_gen_updates = self.opt.get_updates(prior_gen_cost, self.enc_params)
        prior_dis_updates = self.opt.get_updates(prior_dis_cost, self.prior_dis_params)
        data_gen_updates = self.opt.get_updates(data_gen_cost, self.dec_params)
        data_dis_updates = self.opt.get_updates(data_dis_cost, self.data_dis_params)
        ae_updates = self.opt.get_updates(rec_cost, self.enc_params) #+rec_cost +self.dec_params
        
        start_index, end_index = T.iscalars('s_i', 'e_i')
        if self.uint8_data:
            print 'converting uint8 data to float32 for each batch'
            given_train_x = T.cast(self.shared_train[start_index:end_index], dtype='float32')
        else:
            given_train_x = self.shared_train[start_index:end_index]
            
        if self.batch_mean_subtraction:
            assert self.train_mean is not None, 'train_mean cannot be None for batch mean subtraction'
            given_train_x -= self.train_mean
        
        if self.batch_data_process_func is not None:
            given_train_x = self.batch_data_process_func(given_train_x)
        
#         self.get_data_dis_cost = theano.function(
#                               [start_index, end_index, self.z], #
#                               data_dis_show_cost,
#                               givens={self.x:given_train_x}
#                               )
         
        self.train_ae_model = theano.function(
                              [start_index, end_index],
                              rec_show_cost,
                              updates=ae_updates,
                              givens={self.x:given_train_x}
                                )
        
        self.train_data_gen_model = theano.function(
                [start_index, end_index],#[self.z], #
                data_gen_show_cost,
                updates=data_gen_updates,
                givens={self.x:given_train_x}
                )
        
        self.train_data_dis_model = theano.function(
                [start_index, end_index], #, self.z
                data_dis_show_cost,
                updates=data_dis_updates,
                givens={self.x:given_train_x}
                )
        
        self.train_prior_gen_model = theano.function(
            [start_index, end_index],
            prior_gen_show_cost,
            updates=prior_gen_updates,
            givens={self.x:given_train_x}
            )
    
        self.train_prior_dis_model = theano.function(
            [start_index, end_index, self.z],
            prior_dis_show_cost,
            updates=prior_dis_updates,
            givens={self.x:given_train_x}
            )
        

    def get_cost(self, isTest=False, noiseless=False):
        
        # encoding
        encoder_outputs = self.network_fprop(network_layers=self.enc_layers, 
                           x=self.x, 
                           isTest=isTest, 
                           noiseless=noiseless)
        
        # prior generative model
        fake_prior = encoder_outputs[self.encoder_ns[-1]['name']]
        
        dis_fake_prior_outputs = self.network_fprop(network_layers=self.prior_dis_layers, 
                                                    x=fake_prior, 
                                                    isTest=isTest, 
                                                    noiseless=noiseless)
        
        # prior discriminative model
        dis_true_prior_outputs = self.network_fprop(network_layers=self.prior_dis_layers, 
                                                    x=self.z, 
                                                    isTest=isTest, 
                                                    noiseless=noiseless)
        
        ppriorfake = dis_fake_prior_outputs[self.prior_discrim_ns[-1]['name']]
        ppriortrue = dis_true_prior_outputs[self.prior_discrim_ns[-1]['name']]
        
        # get generative cost
        prior_gen_obj = T.mean(T.nnet.binary_crossentropy(ppriorfake, T.ones(ppriorfake.shape)))
        
        # get parameter/regularizer cost
        prior_gen_reg = 0.
        for layer in self.enc_layers:
            prior_gen_reg += layer.getParamRegularization()
        
        # get discriminative cost
        # the discriminative cost is always a binary cross entropy to distinguish
        # between real and model generated samples
        prior_dis_obj = T.mean(T.nnet.binary_crossentropy(ppriorfake, T.zeros(ppriorfake.shape))) \
            + T.mean(T.nnet.binary_crossentropy(ppriortrue, T.ones(ppriortrue.shape)))
        
        # get parameter/regularizer cost
        prior_dis_reg = 0.
        for layer in self.prior_dis_layers:
            prior_dis_reg += layer.getParamRegularization()
        
        #----------------------------------------------------------------------
        # decoding
        
        decoder_outputs = self.network_fprop(network_layers=self.dec_layers, 
                                             x= fake_prior,  # self.z, # 
                                             isTest=isTest, 
                                             noiseless=noiseless)
        
        # data generative model
        dis_fake_data = decoder_outputs[self.decoder_ns[-1]['name']]
        
        
        
        # propogate another time for discriminative objective
        dis_fake_data_outputs = self.network_fprop(network_layers=self.data_dis_layers, 
                                                   x=dis_fake_data,
                                                   isTest=isTest, 
                                                   noiseless=noiseless)
        
        pdatafake = dis_fake_data_outputs[self.data_discrim_ns[-1]['name']]
        # get generative cost
        data_gen_obj = T.mean(T.nnet.binary_crossentropy(pdatafake, T.ones(pdatafake.shape)))
        
        # data discriminative model
        dis_true_data_outputs = self.network_fprop(network_layers=self.data_dis_layers, 
                                                   x=self.x,
                                                   isTest=isTest, 
                                                   noiseless=noiseless)

        
        pdatatrue = dis_true_data_outputs[self.data_discrim_ns[-1]['name']]
        
        # get parameter/regularizer cost
        data_gen_reg = 0.
        for layer in self.enc_layers:
            data_gen_reg += layer.getParamRegularization()
        
        # get discriminative cost
        # the discriminative cost is always a binary cross entropy to distinguish
        # between real and model generated samples
        data_dis_obj = T.mean(T.nnet.binary_crossentropy(pdatafake, T.zeros(pdatafake.shape))) \
            + T.mean(T.nnet.binary_crossentropy(pdatatrue, T.ones(pdatatrue.shape)))
        
        # get parameter/regularizer cost
        data_dis_reg = 0.
        for layer in self.data_dis_layers:
            data_dis_reg += layer.getParamRegularization()
            
        #-----------------------------------------------------------------------
        # reconstruction cost
        reconstruction_outputs = self.network_fprop(network_layers=self.dec_layers, 
                                             x=fake_prior, 
                                             isTest=isTest, 
                                             noiseless=noiseless)
         
        rec_out = reconstruction_outputs[self.decoder_ns[-1]['name']]
 
        rec_obj = self.cost_func.getCost(encoder_outputs[self.data_target_name], 
                                      rec_out)
        rec_reg = 0
        for layer in self.enc_layers+self.dec_layers:
            rec_reg += layer.getParamRegularization()
            
        return (prior_gen_obj+prior_gen_reg, # prior generative cost
                prior_gen_obj, # prior generative show cost
                prior_dis_obj+prior_dis_reg, # piror discriminative cost
                prior_dis_obj, # prior discriminative show cost
                data_gen_obj+data_gen_reg, # data generative cost
                data_gen_obj, # data generative show cost
                data_dis_obj+data_dis_reg, # data discriminative cost
                data_dis_obj, # data discriminative show cost
                rec_obj+rec_reg, # reconstruction cost
                rec_obj)# reconstruction show cost
               
    
    def sample(self, nsamples, noiseless=True):
        gen_outputs = self.network_fprop(network_layers=self.dec_layers, 
                                         x=self.z, 
                                         isTest=True, 
                                         noiseless=noiseless)
        sample_func = theano.function([self.z], gen_outputs[self.decoder_ns[-1]['name']])
        
        nbatches = int(ceil(float(nsamples)/self.batch_size))
        ret = numpy.zeros((nsamples, numpy.prod(self.data_discrim_ns[0]['input_shape'])/self.batch_size), dtype='float32')
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > nsamples:
                batch_end = nsamples
                batch_start = batch_end - self.batch_size
            
            z = self.noise_func(self.batch_size, self.num_z)
            ret[batch_start:batch_end] = sample_func(z).reshape(self.batch_size,-1)
        
        return ret
    
    def sample_from_latent_space(self, zvalues, noiseless=True):
        gen_outputs = self.network_fprop(network_layers=self.dec_layers, 
                                         x=self.z, 
                                         isTest=True, 
                                         noiseless=noiseless)
        sample_func = theano.function([self.z], gen_outputs[self.decoder_ns[-1]['name']])
        nsamples = zvalues.shape[0]
        nbatches = int(ceil(float(nsamples)/self.batch_size))
        ret = numpy.zeros((nsamples, numpy.prod(self.data_discrim_ns[0]['input_shape'])/self.batch_size), dtype='float32')
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > nsamples:
                batch_end = nsamples
                batch_start = batch_end - self.batch_size

            ret[batch_start:batch_end] = sample_func(zvalues[batch_start:batch_end]).reshape(self.batch_size,-1)
        
        return ret
    
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
        
        layer_outputs = self.network_fprop(self.enc_layers, self.x, isTest=True, noiseless=noiseless)
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
        
        layer_outputs = self.network_fprop(self.enc_layers, self.x, isTest=True, noiseless=noiseless)
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
        
        enc_outputs = self.network_fprop(self.enc_layers, self.x, isTest=True, noiseless=noiseless)
        dec_outputs = self.network_fprop(self.dec_layers, enc_outputs[self.encoder_ns[-1]['name']], isTest=True, noiseless=noiseless)
         
        # the reconstruction is always the last layer of the network
        x_hat_given_x = dec_outputs[self.decoder_ns[-1]['name']]
         
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
        
        enc_outputs = self.network_fprop(self.enc_layers, self.x, isTest=True, noiseless=noiseless)
        dec_outputs = self.network_fprop(self.dec_layers, enc_outputs[self.encoder_ns[-1]['name']], isTest=True, noiseless=noiseless)
         
        # the reconstruction is always the last layer of the network
        x_hat_given_x = dec_outputs[self.decoder_ns[-1]['name']]

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
    
    def fit(self, data_provider, train_epoch, optimizer,data_target_name,
            display_func=None, display_freq=-1, show_iteration=-1,
            valid_freq=None, noiseless_validation=True,
            dump_freq=-1, train_mean=None, batch_mean_subtraction=False,
            batch_data_process_func = None,
            verbose=False, *args, **kwargs):
        
        '''fit model to data'''
        assert isinstance(data_provider, UnlabeledDataProvider), (
               'data_provider need to be a subclass from UnlabeledDataProvider'
               ' so that it provides appropriate data for unsupervised models')
        self.data_target_name = data_target_name
        self.data_provider = data_provider
        self.nvalid_batches = 0#data_provider.get_number_of_valid_batches()
        self.shared_train, _, _ = data_provider.get_train_data_and_idx(0)
        self.batch_data_process_func = batch_data_process_func
        
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

        if valid_freq is not None:
            assert valid_freq > 0, 'valid_freq need to be an integer greater than 0'
            valid_freq = valid_freq
        else:
            valid_freq = n_train_batches
            
        print '... training'
        start_time = time.clock()
        
        train_data_dis_model = True
        data_dis_iter = 0
        while (epoch < train_epoch):
            epoch_start = iter_start = time.clock()
            
            epoch = epoch + 1
            data_gen_cost = []
            data_dis_cost = []
            prior_gen_cost = []
            prior_dis_cost = []
            ae_cost = []
            
            for minibatch_index in xrange(n_train_batches):
                it = (epoch - 1) * n_train_batches + minibatch_index
                
                (self.shared_train, batch_start_idx, batch_end_idx) = \
                    data_provider.get_train_data_and_idx(minibatch_index)
                
                

                train_data_dis_model = True
#                 z = self.noise_func(self.batch_size, self.num_z)
#                 minibatch_data_dis_cost = self.get_data_dis_cost(batch_start_idx, batch_end_idx,z)
#                 data_dis_cost.append(minibatch_data_dis_cost)
#                 if minibatch_data_dis_cost < 0.6:
#                     train_data_dis_model = False
                
                #if it % 2 == 1 and train_data_dis_model:
                data_dis_iter += 1
#                 if it%2 != 0:
#                 z = self.noise_func(self.batch_size, self.num_z)
                data_dis_cost.append(self.train_data_dis_model(batch_start_idx, batch_end_idx))
            
                z = self.noise_func(self.batch_size, self.num_z)
                prior_dis_cost.append(self.train_prior_dis_model(batch_start_idx, batch_end_idx, z))
                
                #if it %2 == 0:
#                     rec_cost, gen_cost = self.train_ae_model(batch_start_idx, batch_end_idx)
#                     ae_cost.append(rec_cost)
#                 z = self.noise_func(self.batch_size, self.num_z)
#                 gen_cost = self.train_data_gen_model(z)
                gen_cost = self.train_data_gen_model(batch_start_idx, batch_end_idx)
                data_gen_cost.append(gen_cost)
            
                prior_gen_cost.append(self.train_prior_gen_model(batch_start_idx, batch_end_idx))
            
#                 else:
                for _ in xrange(1):
                    rec_cost = self.train_ae_model(batch_start_idx, batch_end_idx)
                ae_cost.append(rec_cost)

                    
                if it != 0 and dump_freq>0 and it % dump_freq==0:
                    self.dump()
                    
                if display_freq > 0 and display_func is not None and it%display_freq == 0:
                    image = display_func(self.sample(self.batch_size))
                    imsave('%d_gen_image.jpg' % it, image)
                    print 'finished writing sample image'
            
                if ( it != 0 and (show_iteration > 0 and (it+1) % show_iteration==0) 
                    or (show_iteration<0 and (it+1) % n_train_batches==0)):
                    print ('Epoch %d, Iter: %d, Data-D Train Iter: %d\nAE cost: %.4f, Data-G cost: %.4f, Data-D cost: %.4f, '
                           'Prior-G cost: %.4f, Prior-D cost: %.4f' % 
                           (epoch, it, data_dis_iter, numpy.mean(ae_cost), 
                            numpy.mean(data_gen_cost), numpy.mean(data_dis_cost), 
                            numpy.mean(prior_gen_cost), numpy.mean(prior_dis_cost))),
                            
                    data_gen_cost = []
                    data_dis_cost = []
                    prior_gen_cost = []
                    prior_dis_cost = []
                    ae_cost = []
                    print ', time cost: %.4f' % (time.clock() - iter_start)
                    iter_start = time.clock()
            
                if self.nvalid_batches <= 0: # in case of no validation
                    self.copyParamsToBest()
    
                
            print 'Epoch time cost: %.4f' % (time.clock() - epoch_start)
            
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
