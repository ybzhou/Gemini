import theano

import theano.tensor as T

def setupDefaultLayerOptions(pnames, layerSpecs):
    # setup default parameter options
    if 'tune' in layerSpecs:
        tune = layerSpecs['tune']
        # default to tune parameters
        for pn in pnames:
            if pn not in tune:
                tune[pn] = True
    else:
        tune = {}
        # default to tune all parameters
        for pn in pnames:
            tune[pn] = True
            
    if 'reg' in layerSpecs:
        reg = layerSpecs['reg']
        # default to no regularization
        for pn in pnames:
            if pn not in reg:
                reg[pn] = None
    else:
        reg = {}
        # default to no regularization
        for pn in pnames:
            reg[pn] = None
             
    if 'constraint' in layerSpecs:
        constraint = layerSpecs['constraint']
        # default to no constraint
        for pn in pnames:
            if pn not in constraint:
                constraint[pn] = None
    else:
        # default to no constraint
        constraint = {}
        for pn in pnames:
            constraint[pn] = None
    
    # need to have learning rate
    assert 'learning_rate' in layerSpecs, "Learning rate is required"
    lr = layerSpecs['learning_rate']
    
    if 'momentum' in layerSpecs:
        mu = layerSpecs['momentum']
        # default to no momentum
        for pn in pnames:
            if pn not in mu:
                mu[pn] = None
    else:
        # default to no momentum
        mu = {}
        for pn in pnames:
            mu[pn] = None
    
    return tune, reg, constraint, lr, mu

def corrupt(x, theano_rng, noise_type, noise_level):
    if noise_level > 0:
        
        if noise_type == 'saltpepper':
                mask = theano_rng.binomial(size=x.shape, n=1, p=1-noise_level,
                                                         dtype=theano.config.floatX)
                noise = theano_rng.binomial(size=x.shape, n=1, p=0.5,
                                                 dtype=theano.config.floatX)
                return mask*x + T.eq(mask,0)*noise
        elif noise_type == 'normal':
            return theano_rng.normal(size=x.shape, avg=0, std=noise_level, 
                                          dtype=theano.config.floatX) + x
        elif noise_type == 'prod_normal':
            return theano_rng.normal(size=x.shape, avg=1, std=noise_level, 
                                          dtype=theano.config.floatX)*x
        elif noise_type == 'mask':
            return theano_rng.binomial(size=x.shape, n=1, p=1 - noise_level,
                                                dtype=theano.config.floatX) * x
                                                
        elif noise_type == 'col_mask':
            assert(x.ndim==4)
            noise = theano_rng.binomial(size=(x.shape[0], x.shape[2], x.shape[3]), 
                    n=1, p=1 - noise_level,dtype=theano.config.floatX)
            return noise.dimshuffle(0,'x',1,2) * x
        elif noise_type == 'col_normal':
            assert(x.ndim==4)
            noise = theano_rng.normal(size=(x.shape[0], x.shape[2], x.shape[3]), 
                    avg=0, std=noise_level,dtype=theano.config.floatX)
            return noise.dimshuffle(0,'x',1,2) + x
        elif noise_type == 'conv_normal':
            noise = theano_rng.normal(size=x.shape[:2], avg=0, std=noise_level, 
                                          dtype=theano.config.floatX)
            return noise.dimshuffle(0,1,'x','x') + x
        elif noise_type == 'conv_prod_normal':
            noise = theano_rng.normal(size=x.shape[:2], avg=0, std=noise_level, 
                                          dtype=theano.config.floatX)
            return noise.dimshuffle(0,1,'x','x')*x
        elif noise_type == 'conv_mask':
            noise = theano_rng.binomial(size=x.shape[:2], n=1, p=1 - noise_level,
                                                dtype=theano.config.floatX)
            return noise.dimshuffle(0,1,'x','x') * x
        else:
            raise('unkown noise type')
    else:
        return x