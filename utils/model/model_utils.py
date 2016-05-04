
import layers
import pydot
import warnings
import inspect
import sys
import PIL.Image
import theano

# import theano.tensor as T
from activations import Activation

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print "*** Method not implemented: %s at line %s of %s" % (method, line, fileName)
    sys.exit(1)
    
def validate_network(network_structure):
    name_index_dic = {}
    for layer_idx in xrange(len(network_structure)):
        assert network_structure[layer_idx]['name'] not in name_index_dic, "duplicate name %s exist!" % network_structure[layer_idx]['name']
        name_index_dic[network_structure[layer_idx]['name']] = layer_idx
    
    return name_index_dic

def generate_network_plot(network_layers):
    G = pydot.Dot(graph_type='digraph', rankdir='BT')
    for crt_layer in network_layers:
        label_text = crt_layer.layerName
        label_text += '\n'+crt_layer.layerType
        # add hidden layer size
        if isinstance(crt_layer, layers.FullLayer) or isinstance(crt_layer, layers.BidirFullLayer)\
           or isinstance(crt_layer, layers.NormFullLayer):
            label_text += ('\nhiddens: %d' % crt_layer.outputShape[-1]) 
        
        if isinstance(crt_layer, layers.ConvLayer) or isinstance(crt_layer, layers.NormConvLayer):
            # add conv filter size
            label_text += ('\nsize: %d x %d' % (crt_layer.filterShape[-1], crt_layer.filterShape[-1]))
            
            # add number of conv filters
            label_text += ('\nnfilters: %d' % crt_layer.outputShape[1])
        
        if isinstance(crt_layer, layers.PoolLayer):
            # add pool type
            label_text += ('\nmode: %s' % crt_layer.mode)
            
            # add pool shape
            label_text += ('\nsize: %d/%d' % (crt_layer.poolSize, crt_layer.poolStride))
            
        if isinstance(crt_layer, layers.NoiseLayer):
            # add noise type
            label_text += ('\nnoise: %s' % crt_layer.noiseType)
            
            # add noise level
            label_text += ('\nlevel: %.2f' % crt_layer.noiseLevel)
            
        # add activation function TODO wrap all activations to class
        if hasattr(crt_layer, 'actFunc'):
            acttype = 'unknown'
            if crt_layer.actFunc == None:
                acttype = 'linear'
            elif isinstance(crt_layer.actFunc, Activation):
                acttype = crt_layer.actFunc.name
                
            label_text += ('\nact: %s' % acttype)

        G.add_node(pydot.Node(crt_layer.layerName, shape='box', label=label_text))
        
    for crt_layer in network_layers:
        for l in crt_layer.getNextLayer():
            G.add_edge(pydot.Edge(crt_layer.layerName, l.layerName))

    G.write_png('ns_temp.png')
    warnings.warn('A visualization of current network structure is saved as ns_temp.png')
#     img = PIL.Image.open('ns_temp.png')
#     img.load()
#     img.show()
    
def check_network_outputs(network_layers):
    num_layers = len(network_layers)
    for layer_idx in xrange(num_layers):
        # it does not matter if the last layer output is not used by other layers
        # also for the data layer that contain labels
        crt_layer = network_layers[layer_idx]
        if layer_idx < num_layers-1:
            if isinstance(crt_layer, layers.DataLayer):
                if crt_layer.inputType == 'label':
                    continue
            
            # need to have next layer
            assert len(crt_layer.getNextLayer()) > 0, ("Layer %s's output is not used, please check the structure" % crt_layer.layerName)
        

def print_network(network_layers):
    for layer in network_layers:
        outstr = '%s layer -- [%s] takes input from [' % (layer.layerType, layer.layerName)
        
        if isinstance(layer, layers.DataLayer):
            outstr += 'data'
        else:
            prevLayers = layer.getPreviousLayer()
            if len(prevLayers) > 1:
                for prev_layer in prevLayers:
                    outstr += '%s, ' % prev_layer.layerName
            elif len(prevLayers) == 1:
                outstr += prevLayers[0].layerName
            
        outstr += ']'
        
        outstr += ' and output to layer ['
        nextLayers = layer.getNextLayer()
        if len(nextLayers) > 1:
            for next_layer in nextLayers:
                outstr += '%s, ' % next_layer.layerName
        elif len(nextLayers) == 1:
            outstr += nextLayers[0].layerName
        outstr += ']'
        
        outstr += ' input size: ('
        for i in xrange(len(layer.inputShape)):
            if i == len(layer.inputShape)-1:
                outstr += str(layer.inputShape[i])
            else:
                outstr += str(layer.inputShape[i]) + ', '
        outstr += ')'
        
        outstr += ' output size: ('
        for i in xrange(len(layer.outputShape)):
            if i == len(layer.outputShape)-1:
                outstr += str(layer.outputShape[i])
            else:
                outstr += str(layer.outputShape[i]) + ', '
        outstr += ')'
        print outstr

def obtain_network(batch_size,
                   network_structure, name_index_dic, init_params,
                   check_output_usage=True):

    network_layers = []
    
    # first loop for setup prev and next layers for each layer
    crtLayer = None
    for layer_idx in xrange(len(network_structure)):
        crtLayerSpec = network_structure[layer_idx]
        crtLayerSpec['batch_size'] = batch_size
        
        if 'layer_input' in crtLayerSpec:
            crtLayerName = crtLayerSpec['layer_input']
        
        if crtLayerSpec['layer_type'] == 'conv':
            crtLayer = layers.ConvLayer()
        elif crtLayerSpec['layer_type'] == 'pool':
            crtLayer = layers.PoolLayer()
        elif crtLayerSpec['layer_type'] == 'reshape':
            crtLayer = layers.ReshapeLayer()
        elif crtLayerSpec['layer_type'] == 'full':
            crtLayer = layers.FullLayer()
        elif crtLayerSpec['layer_type'] == 'concatenate':
            crtLayer = layers.ConcateLayer()
        elif crtLayerSpec['layer_type'] == 'data':
            crtLayer = layers.DataLayer()
        elif crtLayerSpec['layer_type'] == 'noise':
            crtLayer = layers.NoiseLayer()
        elif crtLayerSpec['layer_type'] == 'bi_full':
            crtLayer = layers.BidirFullLayer()
        elif crtLayerSpec['layer_type'] == 'bi_noise':
            crtLayer = layers.BidirNoiseLayer()
        elif crtLayerSpec['layer_type'] == 'pass':
            crtLayer = layers.PassThroughLayer()
        elif crtLayerSpec['layer_type'] == 'bn_layer':
            crtLayer = layers.BatchNormLayer()
        elif crtLayerSpec['layer_type'] == 'deconv':
            crtLayer = layers.DeConvLayer()
        elif crtLayerSpec['layer_type'] == 'contrast_norm':
            crtLayer = layers.ContrastNormLayer()
        elif crtLayerSpec['layer_type'] == 'dropout':
            crtLayer = layers.DropoutLayer()
        elif crtLayerSpec['layer_type'] == 'norm_full':
            crtLayer = layers.NormFullLayer()
        elif crtLayerSpec['layer_type'] == 'norm_conv':
            crtLayer = layers.NormConvLayer()
        elif crtLayerSpec['layer_type'] == 'stand_layer':
            crtLayer = layers.BatchStandardizeLayer()
        else:
            print 'error'
            raise('Unkown layer type')
        
        if len(network_layers) > 0:
            if 'crtLayerName' in locals() and not isinstance(crtLayer, layers.DataLayer):
                if isinstance(crtLayerName, list):
                    for n in crtLayerName:
                        crtLayer.addPreviousLayer(network_layers[name_index_dic[n]])
                        network_layers[name_index_dic[n]].addNextLayer(crtLayer)
                else:
                    crtLayer.addPreviousLayer(network_layers[name_index_dic[crtLayerName]])
                    network_layers[name_index_dic[crtLayerName]].addNextLayer(crtLayer)
            
        network_layers.append(crtLayer)
        
    # second loop to construct the layer
    for layer_idx in xrange(len(network_structure)):
        crtLayerSpec = network_structure[layer_idx]
        
        if (layer_idx == 0 
            or isinstance(network_layers[layer_idx], layers.DataLayer)):
            # input size needs to be specified for the data layers
            if 'input_shape' in crtLayerSpec:
                if isinstance(crtLayerSpec['input_shape'], tuple):
                    inputSize = (batch_size, ) + crtLayerSpec['input_shape']
                    crtLayerSpec['input_shape'] = inputSize
                elif isinstance(crtLayerSpec['input_shape'], int):
                    inputSize = (batch_size, crtLayerSpec['input_shape'])
                    crtLayerSpec['input_shape'] = inputSize
                else:
                    raise 'input_shape of data layer need to be either a tuple or an integer'
            else:
                # default to batch size
                inputSize = (batch_size, )
                crtLayerSpec['input_shape'] = inputSize
                
        else:
            if isinstance(network_layers[layer_idx], layers.ReshapeLayer):
                if isinstance(crtLayerSpec['shape'], tuple):
                    crtLayerSpec['shape'] = (batch_size,)+crtLayerSpec['shape']
                elif isinstance(crtLayerSpec['shape'], int):
                    crtLayerSpec['shape'] = (batch_size, crtLayerSpec['shape'])
                else:
                    raise 'shape of reshape layer need to be either a tuple or an integer'
            
            # input size will be the previous layer output size
            prevLayer = network_layers[layer_idx].getPreviousLayer()
            # all layers except concatenation layer takes only one input
            # so index 0 is used here, in case of concatenation layer we
            # do not need the input size
            inputSize = prevLayer[0].getOutputShape()
            
        if crtLayerSpec['name'] in init_params:
            network_layers[layer_idx].constructLayer(inputShape=inputSize, 
                                                     initParams=init_params[crtLayerSpec['name']], 
                                                     **crtLayerSpec)
        elif 'share_weight_from_other_layer' in crtLayerSpec:
            assert not isinstance(network_layers[layer_idx], layers.BiDirLayer), \
                "cross layer weight sharing only support feed-forward layers"
            layername, transpose = crtLayerSpec['share_weight_from_other_layer']
            
#             W_expr = network_layers[name_index_dic[layername]].params.getParameter('W')
            W_expr = theano.clone(network_layers[name_index_dic[layername]].params.getParameter('W'),
                                  share_inputs=False)
#             if transpose:
#                 if W_expr.ndim ==  2:
#                     W_expr = W_expr.T
#                 elif W_expr.ndim == 4:
#                     W_expr = W_expr.dimshuffle(1,0,2,3)
#                 else:
#                     raise 'weight should either be a tensor or a matrix'
            network_layers[layer_idx].constructLayer(inputShape=inputSize, 
                                                     initParams=None, 
                                                     weight_outside=(W_expr, transpose),
                                                     **crtLayerSpec)
        else:
            network_layers[layer_idx].constructLayer(inputShape=inputSize, 
                                                     initParams=None, 
                                                     **crtLayerSpec)
        
    # print network connections
    print_network(network_layers)
    
    # visulize network structure
    generate_network_plot(network_layers)
    
    if check_output_usage:
        # check if any layer output is not used
        check_network_outputs(network_layers)
    
    return network_layers