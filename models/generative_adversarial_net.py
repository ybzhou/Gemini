import theano
import layers
import numpy

import time
import copy
import os

import theano.tensor as T
import theano.sandbox.rng_mrg as MRG

from utils.model.model_utils import obtain_network, validate_network
from unsupervised_model import UnsupervisedModel
from math import ceil
from data import UnlabeledDataProvider

__all__ = ['GAN']

class GAN(UnsupervisedModel):
    """ takes network structure and model parameters and assigns class attributes """
    def initilize(self, discriminator_ns, generator_ns, nchannels, image_size, 
                  batch_size, num_z, seed, init_params={}, 
                  *args, **kwargs):
        self.generator_ns = generator_ns
        self.discriminator_ns = discriminator_ns
        self.nchannels = nchannels
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.init_params = init_params
        self.theano_rng = MRG.MRG_RandomStreams(seed)
        self.num_z = num_z
        
    def compile_model(self, *args, **kwargs):
        print '... building the model'
        d_names = [d['name'] for d in self.discriminator_ns]
        g_names = [d['name'] for d in self.generator_ns]
        assert len(set(d_names).intersection(g_names)) == 0, (
        'layer name cannot be the same for generator and discriminator network structure')
        
        total_structure = []
        total_structure.extend(self.generator_ns)
        total_structure.extend(self.discriminator_ns)
        self.name_index_dic = validate_network(total_structure)
        
        self.x = T.matrix('x')
        self.z = T.matrix('z')
        
        self.layers = obtain_network(self.batch_size,
                                     total_structure, 
                                     self.name_index_dic, self.init_params,
                                     check_output_usage=False)
        
        
        # this is to facilitate training
        sep_index = self.name_index_dic[self.discriminator_ns[0]['name']]
        self.gen_layers = self.layers[:sep_index]
        self.dis_layers = self.layers[sep_index:]
        self.gen_params = [l.params for l in self.gen_layers]
        self.dis_params = [l.params for l in self.dis_layers]
        
    def compile_functions(self, opt, **args):
        print '... compiling training functions'
        
        gen_cost, gen_show_cost, dis_cost, cost_pfake, cost_ptrue = self.get_cost()
           
        self.opt = opt
        gen_updates = self.opt.get_updates(gen_cost, self.gen_params)
        dis_updates = self.opt.get_updates(dis_cost, self.dis_params)
           
        self.get_noise = theano.function([],
                                         self.theano_rng.uniform(size=(self.batch_size, self.num_z), 
                                                low=-1, high=1)
                                         ) 
        start_index = T.iscalar('start_index')
        end_index = T.iscalar('end_index')
        
        if self.uint8_data:
            given_train_x = T.cast(self.shared_train[start_index:end_index], dtype='float32')
        else:
            given_train_x = self.shared_train[start_index:end_index]
            
        self.train_gen_model = theano.function(
            [self.z],
            gen_show_cost,
            updates=gen_updates,
            )
        
        self.train_dis_model = theano.function(
            [start_index, end_index, self.z],
            [cost_pfake, cost_ptrue],
            updates=dis_updates,
            givens={self.x: given_train_x}
            )
        

    def get_cost(self, isTest=False):
        
        # generative model
        gen_outputs = self.network_fprop(isGenerator=True, isTest=isTest)
        
        # discriminative model
        dis_fake_outputs = self.network_fprop(isGenerator=False, 
                                              generatedSample=gen_outputs[self.generator_ns[-1]['name']],
                                              isTest=isTest)
        
        dis_true_outputs = self.network_fprop(isGenerator=False, isTest=isTest)
        
        pfake = dis_fake_outputs[self.discriminator_ns[-1]['name']]
        
        ptrue = dis_true_outputs[self.discriminator_ns[-1]['name']]
        
        # get generative cost
        gen_obj = T.mean(T.nnet.binary_crossentropy(pfake, T.ones(pfake.shape)))
        
        # get parameter/regularizer cost
        gen_reg = 0.
        for layer in self.gen_layers:
            gen_reg += layer.getParamRegularization()
        
        # get discriminative cost
        # the discriminative cost is always a binary cross entropy to distinguish
        # between real and model generated samples
        cost_pfake = T.mean(T.nnet.binary_crossentropy(pfake, T.zeros(pfake.shape)))
        cost_ptrue = T.mean(T.nnet.binary_crossentropy(ptrue, T.ones(ptrue.shape)))
        dis_obj = cost_pfake + cost_ptrue
        
        # get parameter/regularizer cost
        dis_reg = 0.
        for layer in self.dis_layers:
            dis_reg += layer.getParamRegularization()
        
        return gen_obj+gen_reg, gen_obj, dis_obj+dis_reg, cost_pfake, cost_ptrue
    
    def network_fprop(self, isGenerator, generatedSample=None, isTest=False):
        if isGenerator:
            fprop_layers = self.gen_layers
        else:
            fprop_layers = self.dis_layers
            
        if isTest:
            mode = 'test'
        else:
            mode = 'train'
            
        layer_outputs = {}
        for layer_idx in xrange(len(fprop_layers)):
            crt_layer = fprop_layers[layer_idx]
            
            if isinstance(crt_layer, layers.DataLayer):
                if crt_layer.inputType == 'data':
                    if generatedSample:
                        layer_outputs[crt_layer.layerName] = crt_layer.fprop(generatedSample)
                    elif isGenerator:
                        # the very first layer of generator takes z
                        layer_outputs[crt_layer.layerName] = crt_layer.fprop(self.z)
                    else:
                        layer_outputs[crt_layer.layerName] = crt_layer.fprop(self.x)
                else:
                    raise('unkown layer input type')
            else:
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
                else:
                    output_for_crt_layer = crt_layer.fprop(input_for_crt_layer)
                
                layer_outputs[crt_layer.layerName] = output_for_crt_layer
                
        return layer_outputs
    
    def sample(self, nsamples):
        gen_outputs = self.network_fprop(isGenerator=True, isTest=True)
        sample_func = theano.function([self.z], gen_outputs[self.generator_ns[-1]['name']])
        
        nbatches = int(ceil(float(nsamples)/self.batch_size))
        ret = numpy.zeros((nsamples, self.nchannels*numpy.prod(self.image_size)), dtype='float32')
        for i in xrange(nbatches):
            batch_start = i*self.batch_size
            batch_end = (i+1)*self.batch_size
            if batch_end > nsamples:
                batch_end = nsamples
                batch_start = batch_end - self.batch_size
            
            code = numpy.asarray(numpy.random.uniform(low=-1, high=1, size=(self.batch_size, self.num_z)), dtype='float32')
            ret[batch_start:batch_end] = sample_func(code).reshape(self.batch_size,-1)
        
        return ret
    
    def extract_feature_from_data_provider(self, data_provider, *args, **kwargs):
        assert False, 'GAN does not learn P(H|X) directly'
        pass
    
    def extract_feature_from_memory_data(self, data, *args, **kwargs):
        assert False, 'GAN does not learn P(H|X) directly'
        pass
    
    def fit(self, data_provider, train_epoch, optimizer, distrain_iter=1,
            dump_freq=-1, train_mean=None, batch_mean_subtraction=False,
            verbose=False, *args, **kwargs):
        
        '''fit model to data'''
        assert isinstance(data_provider, UnlabeledDataProvider), (
               'data_provider need to be a subclass from UnlabeledDataProvider'
               ' so that it provides appropriate data for unsupervised models')
        
        self.data_provider = data_provider
        self.nvalid_batches = data_provider.get_number_of_valid_batches()
        self.shared_train, _, _ = data_provider.get_train_data_and_idx(0)
        
            
        assert self.shared_train.dtype in ['uint8', 'float32'], (
               'shared_train should only have type uint8 or float32')
        self.uint8_data = (self.shared_train.dtype=='uint8')
        
        if train_mean is not None:
            self.train_mean = theano.shared(numpy.asarray(train_mean, dtype='float32'))
        else:
            self.train_mean = None
            
        self.batch_mean_subtraction = batch_mean_subtraction
        
        self.compile_functions(opt=optimizer,
                               *args, **kwargs)

        epoch = 0
        
        # cache for best parameters
        self.best_param_values = {}
        for l in self.layers:
            self.best_param_values[l.layerName] = {}
            layer_param_names = l.params.getAllParameterNames()
            for pn in layer_param_names:
                if not isinstance(l.params.getParameter(pn), T.TensorVariable):
                    self.best_param_values[l.layerName][pn] = copy.deepcopy(l.params.getParameterValue(pn))

        n_train_batches = data_provider.get_number_of_train_batches()
            
        print '... training'
        start_time = time.clock()
        
        while (epoch < train_epoch):
            epoch_start = time.clock()
            
            epoch = epoch + 1
            fcost, tcost = [], []
            gen_cost = []
            for minibatch_index in xrange(n_train_batches):
                it = (epoch - 1) * n_train_batches + minibatch_index
                
                (self.shared_train, batch_start_idx, batch_end_idx) = \
                    data_provider.get_train_data_and_idx(minibatch_index)
                
                z = self.get_noise()
                fc, tc = self.train_dis_model(batch_start_idx, batch_end_idx, z)
                fcost.append(fc)
                tcost.append(tc)
                if it % distrain_iter == 0:
                    z = self.get_noise()
                    gen_cost.append(self.train_gen_model(z))
                
                if it != 0 and dump_freq>0 and it % dump_freq==0:
                    self.dump()

            this_gen_cost = numpy.mean(gen_cost)
            this_fcost = numpy.mean(fcost)
            this_tcost = numpy.mean(tcost) 
            print ('Epoch %d, Total GAN cost: %f, G-cost: %f, True-cost: %f, Fake-cost: %f' 
                   % (epoch, this_gen_cost+this_fcost+this_tcost,
                      this_gen_cost, this_tcost, this_fcost)),

            self.copyParamsToBest()

            print ', time cost: %.4f' % (time.clock() - epoch_start)
            
            
        end_time = time.clock()
        self.dump()
        print('Optimization complete.')
        self.copyBestToParams()
        print ('The code for file ' +
              os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.))
