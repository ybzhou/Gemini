import theano
import layers
import numpy
import warnings

import theano.tensor as T

from utils.model.model_utils import obtain_network, validate_network
from MLP import MLP
from math import ceil

__all__ = ['ECCMLP']

class ECCMLP(MLP):
    
    """ takes network structure and model parameters and assigns class attributes """
    def initilize(self, network_structure,
                  batch_size, seed, network_cost, ecc_code, init_params={}, *args, **kwargs):
        self.network_structure = network_structure
        self.batch_size = batch_size
        self.seed = seed
        self.network_cost = network_cost
        
        # for ECC the labels is required to be vector valued
        for ls  in self.network_structure:
            if ls['layer_type'] == 'data' and ls['input_type'] == 'label':
                assert len(ls['input_shape']) == 1, 'Labels for ECC is required to be vector valued'
                self.label_dims = 2
                
        self.y = T.imatrix('y')
        self.code = theano.shared(value=numpy.asarray(ecc_code, dtype='int32'), 
                                  name='code', borrow=True)
        
        self.init_params = init_params
        
    def get_errors(self):
        return self.custom_error.getError(self.y, self.final_output, self.code)