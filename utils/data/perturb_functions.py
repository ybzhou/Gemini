import numpy
import Image
#from data_utils import cwh_to_whc, whc_to_cwh
import matplotlib.pyplot as plt

_rng = numpy.random

def shuffle_data(data, label=None):
    
    if isinstance(data, list):
        nsamples = data[0].shape[0]
        shuffle_idx = _rng.permutation(nsamples)
        
        # shuffle data
        new_data = []
        for i in xrange(len(data)):
            new_data.append(data[i][shuffle_idx])
            
        if label is None:
            return new_data
        else:
            return new_data, label[shuffle_idx]
    else:
        nsamples = data.shape[0]
        shuffle_idx = _rng.permutation(nsamples)
        
        if label is None:
            return data[shuffle_idx]
        else:
            return data[shuffle_idx], label[shuffle_idx]
        
def shuffle_flip_data(data, label=None):
    if isinstance(data, list):
        nsamples, ndims = data[0].shape
        shuffle_idx = _rng.permutation(nsamples)
        
        # shuffle data
        new_data = []
        img_size = int((ndims) ** 0.5)
        for i in xrange(len(data)):
#             new_d = numpy.zeros(data[i].shape, dtype='float32')
            for idx in xrange(nsamples):
                flat_img = data[i][idx,:]
                img = flat_img.reshape((img_size, img_size))
                if _rng.rand(1) > 0.5:
                    img = img[:,::-1]
                data[i][idx] = img.reshape(-1)
#                 new_d[idx] = img.reshape(-1)
            new_data.append(data[i][shuffle_idx])
            
        if label is None:
            return new_data
        else:
            return new_data, label[shuffle_idx]
        
    else:
        for idx in xrange(data.shape[0]):
            flat_img = data[idx,:]
            img_size = int((flat_img.shape[0]) ** 0.5)
            img = flat_img.reshape((img_size, img_size))
            # flip
            if _rng.rand(1) > 0.5:
                img = img[:,::-1]
            flat_img = img.flatten()
            data[idx,:] = flat_img
            
        nsamples = data.shape[0]
        shuffle_idx = _rng.permutation(nsamples)
        
        return data[shuffle_idx], label[shuffle_idx]

def shuffle_flip_color_data(data, label=None):
    if isinstance(data, list):
        nsamples, ndims = data[0].shape
        shuffle_idx = _rng.permutation(nsamples)
        
        # shuffle data
        new_data = []
        img_size = int((ndims/3) ** 0.5)
        for i in xrange(len(data)):
            new_d = numpy.zeros(data[i].shape, dtype='float32')
            for idx in xrange(nsamples):
                flat_img = data[i][idx,:]
                img = flat_img.reshape((3, img_size, img_size))
                if _rng.rand(1) > 0.5:
                    img = img[:,:,::-1]
                new_d[idx] = img.reshape(-1)
            new_data.append(new_d[shuffle_idx])
            
        if label is None:
            return new_data
        else:
            return new_data, label[shuffle_idx]
        
    else:
        for idx in xrange(data.shape[0]):
            flat_img = data[idx,:]
            img_size = int((flat_img.shape[0]/3) ** 0.5)
            img = flat_img.reshape((3, img_size, img_size))
            img = cwh_to_whc(img)
            img = Image.fromarray(img)
            # flip
            if numpy.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = numpy.asarray(img)
            img = whc_to_cwh(img)
            flat_img = img.flatten()
            data[idx,:] = flat_img
            
        nsamples = data.shape[0]
        shuffle_idx = _rng.permutation(nsamples)
        
        return data[shuffle_idx], label[shuffle_idx]

def shuffle_crop_color_data(data, label):
    if isinstance(data, list):
        nsamples, ndims = data[0].shape
        shuffle_idx = _rng.permutation(nsamples)
        
        # shuffle data
        new_data = []
        img_size = int((ndims/3) ** 0.5)
        border = 4
        for i in xrange(len(data)):
            new_d = numpy.zeros((data[i].shape[0], 3*(img_size-2*border)**2), dtype='float32')
            for idx in xrange(nsamples):
                flat_img = data[i][idx,:]
                img = flat_img.reshape((3, img_size, img_size))
                if _rng.rand(1) > 0.5:
                    img = img[:,:,::-1]
                    
                x = _rng.randint(2*border+1)
                y = _rng.randint(2*border+1)
                
                img = img[:,y:y+img_size-2*border, x:x+img_size-2*border]
                
                new_d[idx] = img.reshape(-1)
            new_data.append(new_d[shuffle_idx])
            
        if label is None:
            return new_data
        else:
            return new_data, label[shuffle_idx]
        
    else:
        crop_pct = 0.9
        for idx in xrange(data.shape[0]):
            flat_img = data[idx,:]
            img_size = int((flat_img.shape[0]/3) ** 0.5)
            img = flat_img.reshape((3, img_size, img_size))
            crop_size = int(crop_pct * img_size)
            max_x = img_size - crop_size
            max_y = img_size - crop_size
            x = numpy.random.choice(numpy.arange(0,max_x+1,1))
            y = numpy.random.choice(numpy.arange(0,max_y+1,1))
            # crop
            img = img[:,x:x+crop_size,y:y+crop_size]
            img = cwh_to_whc(img)
            img = Image.fromarray(img)
            img = img.resize((img_size,img_size),Image.ANTIALIAS)
    #         plt.imshow(img);plt.show()
            img = numpy.asarray(img)
            img = whc_to_cwh(img)
            flat_img = img.flatten()
            data[idx,:] = flat_img
            
        nsamples = data.shape[0]
        shuffle_idx = _rng.permutation(nsamples)
        
        return data[shuffle_idx], label[shuffle_idx]