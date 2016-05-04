'''
This module serves as a wrpper for basic functionalities of theano so that
it is easier to switch from theano to tensorflow
'''

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as mrg_rng
import numpy
import inspect
from theano.tensor.signal import pool

from utils.common import FLOAT_TYPE, EPSILON

theano.config.floatX = FLOAT_TYPE

# ------------------------------------------------------------------------------
# rng
class RNG(object):
    def __init__(self, seed=None):
        if seed is None:
            seed = numpy.random.randint(2**30)
        
        self.rng = mrg_rng.MRG_RandomStreams(seed=seed)
        
    def binomial(self, size, n=1, p=0.0, dtype=FLOAT_TYPE):
        return self.rng.binomial(size=size, n=n, p=p, dtype=dtype)
    
    def normal(self, size, avg=0.0, std=1.0, dtype=FLOAT_TYPE):
        return self.rng.normal(size=size, avg=mean, std=std, dtype=dtype)
    
    def uniform(self, size, low=0.0, high=1.0, dtype=FLOAT_TYPE):
        return self.rng.uniform(size=size, low=low, high=high, dtype=dtype)

# ------------------------------------------------------------------------------
# Variables

def data_variable(value, dtype=FLOAT_TYPE, name=None):
    '''create a variable that holds data'''
    np_arr = numpy.asarray(value, dtype=dtype)
    return theano.shared(value=np_arr, name=name)

def symbolic_variable(name=None, dtype=FLOAT_TYPE, shape=None, ndims=None):
    '''create a symbolic variable for computation'''
    assert not(shape is None and ndims is None), 'Need to provide either shape or ndims'
    
    if shape is not None:
        ndims = len(shape)
    
    broadcast_dim = (False, )*ndims
    return T.TensorType(dtype=dtype, broadcastable=broadcast_dim)(name)

# ------------------------------------------------------------------------------
# Attributes

# def type(x):
#     return x.type

def ndim(x):
    return x.ndim

def dtype(x):
    return x.dtype

def reshape(x, shape):
    return x.reshape(shape)

def shape(x):
    return x.shape

def dimshuffle(x, *pattern):
    return x.dimshuffle(*pattern)

def flatten(x, ndim=1):
    return x.flatten(ndim)

def nonzero(x, return_matrix=False):
    return x.nonzero(return_matrix=return_matrix)

def sort(x, axis=-1, kind='quicksort', order=None):
    return x.sort(axis=axis, kind=kind, order=order)

def argsort(x, axis=-1, kind='quicksort', order=None):
    return x.argsort(axis=axis, kind=kind, order=order)

def get_value(x):
    return x.get_value()

def set_value(x, value):
    x.set_value(numpy.asarray(value, dtype=x.dtype))
    
# ------------------------------------------------------------------------------
# Function wrapper

def any(x, axis=None, keepdims=False):
    return T.any(x, axis=axis, keepdims=keepdims)

def all(x, axis=None, keepdims=False):
    return T.all(x, axis=axis, keepdims=keepdims)

def sum(x, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    return T.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype)

def prod(x, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    return T.prod(x, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype)

def mean(x, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    return T.mean(x, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype)

def var(x, axis=None, keepdims=False):
    return T.var(x, axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)

def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)

def argmin(x, axis=None, keepdims=False):
    return T.argmin(x, axis=axis, keepdims=keepdims)

def argmax(x, axis=None, keepdims=False):
    return T.argmax(x, axis=axis, keepdims=keepdims)

def clone(x, share_inputs=True):
    return theano.clone(x, share_inputs=share_inputs)

def addbroadcast(x, *axes):
    return T.addbroadcast(x, *axes)

def arange(start, stop=None, step=1, dtype=None):
    return T.arange(start, stop, step, dtype)

# creating tensor

def zeros_like(x):
    return T.zeros_like(x)

def ones_line(x):
    return T.ones_like(x)

def ones(shape, dtype=None):
    return T.ones(shape=shape, dtype=dtype)

def zeros(shape, dtype=None):
    return T.zeros(shape, dtype=None)

def identity_like(x):
    return T.identity_like(x)

def concatenate(tensor_list, axis=0):
    return T.concatenate(tensor_list, axis=axis)

# casting

def cast(x, dtype):
    return T.cast(x, dtype=dtype)

# comparisons
def lt(a, b):
    return T.lt(a, b)

def gt(a, b):
    return T.gt(a, b)

def le(a, b):
    return T.le(a, b)

def ge(a, b):
    return T.ge(a, b)

def eq(a, b):
    return T.eq(a, b)

def neq(a, b):
    return T.neq(a, b)

def isnan(x):
    return T.isnan(x)

def isinf(x):
    return T.isinf(x)

# condition
def switch(cond, ift, iff):
    return T.switch(cond=cond, ift=ift, iff=iff)

def ifelse(cond, then_branch, else_branch, name=None):
    return theano.ifelse.ifelse(cond, then_branch, else_branch, name)

def clip(x, a_min, a_max):
    return T.clip(x, a_min=a_min, a_max=a_max)

# bitwise

def and_(a, b):
    return T.and_(a, b)

def or_(a, b):
    return T.or_(a, b)

def xor(a, b):
    return T.xor(a, b)

def invert(x):
    return T.invert(x)

# math

def abs(x):
    return T.abs_(x)

def angle(x):
    return T.angle(x)

def exp(x):
    return T.exp(x)

def maximum(a, b):
    return T.maximum(a, b)

def minimum(a, b):
    return T.minimum(a, b)

def neg(x):
    return T.neg(x)

def inv(x):
    return T.inv(x)

def log(x):
    return T.log(x)

def log2(x):
    return T.log2(x)

def log10(x):
    return T.log10(x)

def sign(x):
    return T.sgn(x)

def ceil(x):
    return T.ceil(x)

def floor(x):
    return T.floor(x)

def round(x):
    return T.round(x)

def square(x):
    return T.sqr(x)

def pow(x, d):
    return T.pow(x, d)

def sqrt(x):
    return T.sqrt(x)

def cos(x):
    return T.cos(x)

def sin(x):
    return T.sin(x)

def tan(x):
    return T.tan(x)

def cosh(x):
    return T.cosh(x)

def sinh(x):
    return T.sinh(x)

# linear algebra

def transpose(x):
    return T.transpose(x)

def dot(x, y):
    return T.dot(x, y)

def outer(x, y):
    return T.outer(x, y)

def tensordot(x, y, axes=2):
    return T.tensordot(x, y, axes)

# gradient

def grad(cost, wrt):
    return T.grad(cost, wrt)

# nnet ops
def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def softplus(x):
    return T.nnet.softplus(x)

def softmax(x):
    return T.nnet.softmax(x)

def relu(x, alpha=0):
    return T.nnet.relu(x, alpha)

def binary_cross_entropy(output, target):
    return T.nnet.binary_crossentropy(output, target)

def categorical_cross_entropy(output, target):
    return T.nnet.categorical_crossentropy(output, target)

def conv2d(x, filter, input_shape=None, filter_shape=None, border_mode='valid',
           stride=(1, 1)):
    '''
    Perform 2d convolution
    
    Parameters
    ----------
    x : a symbolic 4-D tensor 
        This parameter stands for the input.
    filter : a symbolic 4-D tensor variable
        This parameter stands for the filter for convolution.
    input_shape : {None, tuple/list of length 4)
        The shape of the input, i.e. x, this parameter is optional. By 
        providing this parameter, the function may choose to use it to 
        determine the optimal implementation.
    filter_shape : {None, tuple/list of length 4)
        The shape of the filter, i.e. paramter filter, this parameter is 
        optional. By providing this parameter, the function may choose to 
        use it to determine the optimal implementation.
    border_mode : {'same', 'valid', 'full', int, (int, int)}
        This parameter specifies the border mode for convolution when the 
        provided values are from {'same', 'valid', 'full'}, and it specifies 
        the padding when the provided value is an int or a tuple of two ints.
        
        ``'valid'``: only apply filter when the filter can be completely 
            overlapped with the input. The resultant output shape is: 
            input shape - filter shape + 1
        
        ``'same'``: as the name suggests, it will return the ouput with the 
            same spatial dimension as the input. This is equivalent as padding 
            the input with ``filter width / 2`` to the columns of the input and 
            ``filter height / 2`` to the rows of the input and do valid 
            convolution.
            
        ``'full'``: apply the filter whenever there is an overlap with the input.
            The output size is: input shape + filter shape - 1
            
        ``int``: zero padding the input with the specified value on all sides, 
            and then perform valid convolution.
        
        ``(int1, int2)``: pad input with ``int1`` rows and ``int2`` columns, 
            and then perform valid convolution.
            
    stride : int tuple of length 2
    '''
    if (border_mode not in ['valid', 'same', 'full']  
        and type(border_mode) is not int
        and type(border_mode) is not tuple):
        raise Exception("border_mode needs to be either of the following:\n"
                        +"\t{'valid', 'same', 'full'}\n"
                        +"\tint value specifies the symmetric padding on the border"
                        +"\t(int, int) value specifies the padding on borders")
    
    if border_mode == 'same':
        border_mode = 'half'
    
    conv_out = T.nnet.conv2d(x, filter,
                             border_mode=border_mode,
                             subsample=stride,
                             input_shape=input_shape,
                             filter_shape=filter_shape)
    return conv_out

def pool2d(x, pool_size, stride=None, padding=(0, 0),
           mode='max'):
    '''
    Perform 2d pooling
    
    Parameters
    ----------
    x : a symbolic 4-D tensor 
        This parameter stands for the input.
    pool_size : an int tuple of length 2
        This parameter stands for size of the pooling window
    stride : {None, tuple of length 2}
        The stride of pooling operation, when set to None the stride is equal 
        to pool_size
    padding : an int tuple of length 2
        This parameter specifies the padding for pooling.
        ``(int1, int2)``: pad input with ``int1`` rows and ``int2`` columns, 
            and then perform pooling.
    mode : {'max', 'avg'}
        This parameter specifies the mode of pooling.
    '''
    if type(pool_size) is not tuple and len(pool_size) != 2:
        raise Exception("pool_size needs to be a int tuple of length 2")
    
    if stride is not None and type(stride) is not tuple and len(stride)!=2:
        raise Exception("stride needs to be a int tuple of length 2 or "+
                        "None")
        
    if mode not in ['max', 'avg']:
        raise Exception("mode nees to take value from 'max' or 'avg'")
    
    if stride is None:
        stride = pool_size

    if mode == 'avg':
        mode = 'average_exc_pad'
    
    pool_out = pool.pool_2d(x, ds=pool_size,
                            ignore_border=True, 
                            st=stride,
                            padding=padding,
                            mode=mode)
    
    return pool_out

# computation graph

def evaluate(x):
    '''eval function'''
    return x.eval()

# the following is taken from keras (http://keras.io/)
class Function(object):
    def __init__(self, inputs, outputs, updates=None, givens=None, **kwargs):
        self.function = theano.function(inputs=inputs, 
                                        outputs=outputs, 
                                        updates=updates,
                                        givens=givens,
                                        **kwargs)

    def __call__(self, *inputs):
        assert type(inputs) in {list, tuple}
        return self.function(*inputs)


def function(inputs, outputs, updates=None, givens=None, **kwargs):
    if len(kwargs) > 0:
        function_args = inspect.getargspec(theano.function)[0]
        for key in kwargs.keys():
            if key not in function_args:
                msg = "Invalid argument '%s' passed to function" % key
                raise ValueError(msg)
    return Function(inputs=inputs, 
                    outputs=outputs, 
                    updates=updates, 
                    givens=givens,
                    **kwargs)

