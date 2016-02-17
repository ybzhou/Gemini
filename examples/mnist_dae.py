import theano
import cPickle
import numpy
import sys
import hickle
import theano.tensor as T
import theano.sandbox.rng_mrg as MRG_RNG

import utils.model.optimizers as optimizers
import utils.model.schedulers as schedulers
import utils.model.weight_init as weight_init

from math import sqrt
from models import AutoEncoder
from models import MLP
from utils.model.costs import SoftmaxCost, BinaryCrossEntropyCost
from utils.model.regularizers import L2Regularizer
from utils.model.constraints import MaxColNormConstraint, L2ColNormConstraint
from utils.image.tile_raster_image import tile_raster_images
from PIL import Image
# from models.ForwardAE import ForwardAE
from data import MNISTDataLoader
from data import UnlabeledMemoryDataProvider, LabeledMemoryDataProvider
from utils.model.activations import *

seed = 1234
numpy.random.seed(seed)
numpy_rng = numpy.random.RandomState(seed)
theano_rng = MRG_RNG.MRG_RandomStreams(numpy_rng.randint(2 ** 30))
pretrain_epochs = 100
finetune_epochs = 100
batch_size = 100
max_gpu_samples = 50000

nchannels = 1
image_size = (28,28)
update_freq = 500000
LR_start = 0.001
alpha = 0.1
LR_fin = 0.001/100.
L_decay = (LR_fin/LR_start)**(1./finetune_epochs)

ae_ns = [
    {'name': 'data1', 'layer_type':'data', 'input_type':'data','input_shape':(numpy.prod(image_size))},
    
    {'name': 'noise1', 'layer_input':'data1', 'layer_type':'noise', 
    'noise_type':'normal', 'noise_level':0.7, 'theano_rng':theano_rng},
          
    {'name':'full1', 'layer_input':'noise1', 'layer_type':'full', 'hiddens':1000, 
     'w_init':weight_init.NormalizedWeightInit(numpy_rng), 'b_init':0, 'act_func':Sigmoid(),
     'constraint':{'W':MaxColNormConstraint(3.86)},
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)},
    
    {'name': 'noise2', 'layer_input':'full1', 'layer_type':'noise', 
    'noise_type':'normal', 'noise_level':0.7, 'theano_rng':theano_rng},
          
    {'name':'full2', 'layer_input':'noise2', 'layer_type':'full', 'hiddens':1000, 
     'w_init':weight_init.NormalizedWeightInit(numpy_rng), 'b_init':0, 'act_func':Sigmoid(),
     'constraint':{'W':MaxColNormConstraint(3.86)},
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)},
          
    {'name':'rec_full2', 'layer_input':'full2', 'layer_type':'full', 'hiddens':1000, 
     'w_init':None, 'b_init':0., 'act_func':Sigmoid(),
     'tune':{'W':False, 'b':True},
     'share_weight_from_other_layer': ('full2', True),
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)}, 
    
    {'name':'rec', 'layer_input':'rec_full2', 'layer_type':'full', 'hiddens':numpy.prod(image_size), 
     'w_init':None, 'b_init':0., 'act_func':Sigmoid(),
     'tune':{'W':False, 'b':True},
     'share_weight_from_other_layer': ('full1', True),
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)}, 
]

ae_cost = {'rec':('data1', BinaryCrossEntropyCost())}

mlp_ns = [
    {'name': 'label', 'layer_type':'data', 'input_type':'label'},
    {'name': 'data1', 'layer_type':'data', 'input_type':'data','input_shape':(numpy.prod(image_size))},
    
    {'name':'full1', 'layer_input':'data1', 'layer_type':'full', 'hiddens':1000, 
     'w_init':None, 'b_init':0, 'act_func':Sigmoid(),
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)},
    
    {'name':'full2', 'layer_input':'full1', 'layer_type':'full', 'hiddens':1000, 
     'w_init':None, 'b_init':0, 'act_func':Sigmoid(),
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)},
          
    {'name':'output', 'layer_input':'full2', 'layer_type':'full', 'hiddens':10, 
     'w_init':weight_init.NormalizedWeightInit(numpy_rng), 'b_init':0., 'act_func':Softmax(),
     'learning_rate':{'W':LR_start, 'b':LR_start},
     'lr_scheduler':schedulers.AnnealScheduler(update_freq, L_decay, 1e-7)}, 
]

mlp_cost = {'output':('label', SoftmaxCost())}

def test_mnist_ae():
    dl = MNISTDataLoader('/data/Research/datasets/mnist/mnist.pkl')
    
    dp = UnlabeledMemoryDataProvider(data_loader=dl, 
                                         batch_size=batch_size, 
                                         max_gpu_train_samples=max_gpu_samples, 
                                         max_gpu_valid_samples=max_gpu_samples, 
                                         is_test=False, 
                                         epochwise_shuffle=False, 
                                         nvalid_samples=0)
    
    opt = optimizers.SGD_Rms_Optimizer(decay=0.9)
    
    ae = AutoEncoder(batch_size = batch_size, seed=seed,
            network_structure=ae_ns, network_cost=ae_cost)
    
    ae.fit(data_provider=dp, 
            optimizer=opt,
            train_epoch=pretrain_epochs,
            noiseless_validation=True,
            dump_freq=(60000//100+1)*10)
     
    ae.save('testmodel_ae.joblib')
#     ae.load('testmodel_ae.joblib')

    Image.fromarray(tile_raster_images(ae.params[2].getParameterValue('W').T, (28,28), (10, 10))).show()   

    f = open('/data/Research/datasets/mnist/mnist.pkl', 'rb')
    _, _, test_set = cPickle.load(f)
    f.close()
    
    dp = UnlabeledMemoryDataProvider(data_loader=dl, 
                                         batch_size=batch_size, 
                                         max_gpu_train_samples=max_gpu_samples, 
                                         max_gpu_valid_samples=max_gpu_samples, 
                                         is_test=True, 
                                         epochwise_shuffle=False, 
                                         nvalid_samples=0)
    
    feat1 = ae.extract_feature_from_memory_data(test_set[0], 'full2', 1, True)
    feat2 = ae.extract_feature_from_data_provider(dp, 'full2', None, False, 1, True)
    
    print numpy.all(numpy.equal(feat1, feat2))
    
    rec1 = ae.reconstruct_from_memory_data(test_set[0][:9])
    Image.fromarray(tile_raster_images(rec1, (28,28), (3, 3))).show()
    rec2 = ae.reconstruct_from_data_provider(dp)
    Image.fromarray(tile_raster_images(rec2, (28,28), (3, 3))).show()
    print numpy.all(numpy.equal(rec1, rec2[:9]))
    
    rec1 = ae.reconstruct_from_memory_data(test_set[0][:9], steps=10, noiseless=False)
    Image.fromarray(tile_raster_images(rec1, (28,28), (3, 3))).show()
    
def test_finetune():
    
    
    dl = MNISTDataLoader('/data/Research/datasets/mnist/mnist.pkl')
    
    dp = LabeledMemoryDataProvider(data_loader=dl, 
                                         batch_size=batch_size, 
                                         max_gpu_train_samples=max_gpu_samples, 
                                         max_gpu_valid_samples=max_gpu_samples, 
                                         is_test=False, 
                                         epochwise_shuffle=False, 
                                         nvalid_samples=10000)

    opt = optimizers.SGD_Rms_Optimizer()
    
    init_params = hickle.load('testmodel_ae.joblib')
    
    mlp = MLP(batch_size = batch_size, seed=seed,
            network_structure=mlp_ns, network_cost=mlp_cost,
            init_params=init_params)
    
    mlp.fit(data_provider=dp, 
            optimizer=opt,
            train_epoch=finetune_epochs, 
            early_stop=False, 
            dump_freq=(50000//100+1)*10)
      
    mlp.save('mnist_mlp_model.joblib')
    
    f = open('/data/Research/datasets/mnist/mnist.pkl', 'rb')
    _, _, test_set = cPickle.load(f)
    f.close()
    
    pred = mlp.predict_from_memory_data(test_set[0])
    print 'Error: ', numpy.mean(pred!=test_set[1])
    
    
    dp = LabeledMemoryDataProvider(data_loader=dl, 
                                     batch_size=batch_size, 
                                     max_gpu_train_samples=max_gpu_samples, 
                                     max_gpu_valid_samples=max_gpu_samples, 
                                     is_test=True, 
                                     epochwise_shuffle=False, 
                                     nvalid_samples=0)
    pred = mlp.predict_from_data_provider(dp)
    print 'Error: ', numpy.mean(pred!=test_set[1])

if __name__ == '__main__':
    test_mnist_ae()
    test_finetune()