import cPickle
import numpy
import theano.tensor as T
import theano.sandbox.rng_mrg as MRG_RNG

import utils.model.optimizers as optimizers
import utils.model.schedulers as schedulers
import utils.model.weight_init as weight_init

from utils.model.costs import SoftmaxCost
from utils.model.regularizers import L2Regularizer
from models import MLP
from data import LabeledMemoryDataProvider
from data import CIFAR10DataLoader

seed = 1234
numpy.random.seed(seed)
numpy_rng = numpy.random.RandomState(seed)
theano_rng = MRG_RNG.MRG_RandomStreams(numpy_rng.randint(2 ** 30))
training_epochs = 20
batch_size = 100
max_gpu_samples = 50000

nchannels = 3
image_size = (32,32)
update_freq = 50000/batch_size*10
L_decay = 0.1

mlp_ns = [
    {'name': 'label', 'layer_type':'data', 'input_type':'label'},
    {'name': 'data1', 'layer_type':'data', 'input_type':'data','input_shape':(nchannels, image_size[0], image_size[1])},
    
    {'name':'conv1', 'layer_input':'data1', 'layer_type':'conv', 'channels':32, 
     'strid_size':1, 'pad':2, 'filter_size':5,
     'w_init':weight_init.GaussianWeightInit(numpy_rng, 0.0001), 'b_init':0.,
     'reg':{'W':L2Regularizer(0.004)}, #, 'b':None
     'learning_rate':{'W':0.001, 'b':0.002},
     'momentum':{'W':0.9, 'b':0.9},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq=update_freq, anneal_coef=L_decay, min_rate=1e-7)},
          
    {'name':'pool1', 'layer_input':'conv1', 'layer_type':'pool', 
     'pool_size':3, 'pool_stride':2, 'pool_mode':'max', 'act_func':T.nnet.relu},
    
    {'name':'conv2', 'layer_input':'pool1', 'layer_type':'conv', 'channels':32, 
     'strid_size':1, 'pad':2, 'filter_size':5,
     'w_init':weight_init.GaussianWeightInit(numpy_rng, 0.01), 'b_init':0., 'act_func':T.nnet.relu,
     'reg':{'W':L2Regularizer(0.004)},
     'learning_rate':{'W':0.001, 'b':0.002},
     'momentum':{'W':0.9, 'b':0.9},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq=update_freq, anneal_coef=L_decay, min_rate=1e-7)},
          
    {'name':'pool2', 'layer_input':'conv2', 'layer_type':'pool', 
     'pool_size':3, 'pool_stride':2, 'pool_mode':'average', 'act_func':None},
          
    {'name':'conv3', 'layer_input':'pool2', 'layer_type':'conv', 'channels':64, 
     'strid_size':1, 'pad':2, 'filter_size':5,
     'w_init':weight_init.GaussianWeightInit(numpy_rng, 0.01), 'b_init':0., 'act_func':T.nnet.relu,
     'reg':{'W':L2Regularizer(0.004)},
     'learning_rate':{'W':0.001, 'b':0.002},
     'momentum':{'W':0.9, 'b':0.9},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq=update_freq, anneal_coef=L_decay, min_rate=1e-7)},
          
    {'name':'pool3', 'layer_input':'conv3', 'layer_type':'pool', 
     'pool_size':3, 'pool_stride':2, 'pool_mode':'average', 'act_func':None},
          
    {'name':'full1', 'layer_input':'pool3', 'layer_type':'full', 'hiddens':64, 
     'w_init':weight_init.GaussianWeightInit(numpy_rng, 0.1), 'b_init':0, 'act_func':T.nnet.relu,
     'reg':{'W':L2Regularizer(0.03)},
     'learning_rate':{'W':0.001, 'b':0.002},
     'momentum':{'W':0.9, 'b':0.9},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq=update_freq, anneal_coef=L_decay, min_rate=1e-7)},
    
    {'name':'output', 'layer_input':'full1', 'layer_type':'full', 'hiddens':10, 
     'w_init':weight_init.GaussianWeightInit(numpy_rng, 0.1), 'b_init':0, 'act_func':T.nnet.softmax,
     'tune':{'W':True, 'b':True}, 'reg':{'W':L2Regularizer(0.03)},
     'learning_rate':{'W':0.001, 'b':0.002},
     'momentum':{'W':0.9, 'b':0.9},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq=update_freq, anneal_coef=L_decay, min_rate=1e-7)},
]

mlp_cost = {'output':('label', SoftmaxCost())}


def test_cifar():
    print 'loading data'
    
    dl = CIFAR10DataLoader(cifar_file_path='/data/Research/datasets/cifar10/cifar-10-batches-py',
                           load_dtype='float32')
    
    dp = LabeledMemoryDataProvider(data_loader=dl, 
                                         batch_size=batch_size, 
                                         max_gpu_train_samples=max_gpu_samples, 
                                         max_gpu_valid_samples=max_gpu_samples, 
                                         is_test=False, 
                                         epochwise_shuffle=False, 
                                         nvalid_samples=5000)

    opt = optimizers.SGD_Momentum_Optimizer()
    
    mlp = MLP(batch_size = batch_size, seed=seed,
            network_structure=mlp_ns, network_cost=mlp_cost)
    
    mlp.fit(data_provider=dp, 
            optimizer=opt,
            train_epoch=training_epochs, 
            early_stop=10, 
            dump_freq=(50000//100+1)*10,
            train_mean=dp.get_data_stats(),
            batch_mean_subtraction=True)
      
    mlp.save('alex_cifar_26_model.joblib')
#     mlp.load('alex_cifar_26_model.joblib') 
#     
    f = open('/data/Research/datasets/cifar10/cifar-10-batches-py/test_batch', 'rb')
    data = cPickle.load(f)
    f.close()
    test_set = data['data']
    test_y = data['labels']
     
    test_set = numpy.asarray(test_set, dtype='float32')
    tm = dp.get_data_stats()
    test_set -= tm
     
    pred = mlp.predict_from_memory_data(test_set)
    print 'Error: ', numpy.mean(pred!=test_y)
    
    
    dp = LabeledMemoryDataProvider(data_loader=dl, 
                                         batch_size=batch_size, 
                                         max_gpu_train_samples=max_gpu_samples, 
                                         max_gpu_valid_samples=max_gpu_samples, 
                                         is_test=True, 
                                         epochwise_shuffle=False, 
                                         nvalid_samples=0)
    pred = mlp.predict_from_data_provider(dp, train_mean=tm, batch_mean_subtraction=True)
    print 'Error: ', numpy.mean(pred!=test_y)
    
def write_cifar():
    for i in xrange(5):
        f = open('/data/Research/datasets/cifar10/cifar-10-batches-py/data_batch_%d' % (i+1), 'rb')
        data = cPickle.load(f)
        f.close()
        if i == 0:
            train_set = data['data']
            train_y = data['labels']
        else:
            train_set = numpy.vstack((train_set, data['data']))
            train_y = numpy.hstack((train_y, data['labels']))
            
    import hickle
    hickle.dump((train_set, train_y), '/data/Research/datasets/cifar10/cifar-10-batches-py/train.hic',
                compression='gzip')

    f = open('/data/Research/datasets/cifar10/cifar-10-batches-py/test_batch', 'rb')
    data = cPickle.load(f)
    f.close()
    test_set = data['data']
    test_y = data['labels']
    
    hickle.dump((test_set, test_y), '/data/Research/datasets/cifar10/cifar-10-batches-py/test.hic',
                compression='gzip')
if __name__ == '__main__':
#     write_cifar()
    test_cifar()