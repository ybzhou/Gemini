
FLOAT_TYPE = 'float32'
EPSILON = 1e-7

def get_epsilon():
    '''return the current global epsilon value'''
    return EPSILON

def set_epsilon(eps):
    '''set the global epsilon value'''
    global EPSILON
    EPSILON = eps
    
def get_float_precision():
    '''return the current global float precision flag'''
    return FLOAT_TYPE

def set_float_precision(fp):
    '''set the global float precision flag'''
    global FLOAT_TYPE
    if fp not in ['float32', 'float64']:
        raise Exception('Unkown float type: ' + str(fp))
    
    FLOAT_TYPE = str(fp)