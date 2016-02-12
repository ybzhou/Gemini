from data_provider import UnlabeledDataProvider

import hickle
import numpy
import theano
import warnings
from data_loader import DataLoader

from math import ceil


__all__ = ['UnlabeledDataProvider']

class UnlabeledMemoryDataProvider(UnlabeledDataProvider):
    
    '''
       LabeledMemoryDataProvider assumes the full data can fit in the memory of the
       current computer, it is best suited for small to medium sized datasets.
       The data is required to be saved in a hickle file.
       Loading dataset that cannot fit into the memory may cause memory
       error.
    '''
    
    def __init__(self, data_loader, 
                 batch_size = 100,
                 max_gpu_train_samples=10000,
                 max_gpu_valid_samples=10000,
                 is_test=False,
                 epochwise_shuffle=False,
                 perturb_function=None,
                 nvalid_samples=0):
        
        assert nvalid_samples >= 0, 'nvalid_samples need to be a non-negative integer'
        assert max_gpu_train_samples%batch_size==0, 'max_gpu_train_samples need to be multiple of batch_size'
        assert max_gpu_valid_samples%batch_size==0, 'max_gpu_valid_samples need to be multiple of batch_size'
        assert not is_test or nvalid_samples==0, 'nvalid_samples has to be 0 during test'
        assert isinstance(data_loader, DataLoader), 'need to provide data_loader which is a subclass of DataLoader'
        
        super(UnlabeledMemoryDataProvider, self).__init__(
                                    data_loader=data_loader, 
                                    batch_size=batch_size,
                                    is_test=is_test,
                                    nvalid_samples=nvalid_samples)
        
        self.epochwise_shuffle = epochwise_shuffle
        self.perturb_function = perturb_function
        
        # in case there is no validation data from the data loader
        if data_loader.get_nvalid_data() <= 0:
            if is_test:
                data = data_loader.get_test_data()
            else:
                data = data_loader.get_train_data()

        else:
            data = numpy.vstack((data_loader.get_train_data(), data_loader.get_valid_data()))
            nvalid_samples = data_loader.get_nvalid_data()
            
        nsamples = data.shape[0]
        ntrain_samples = nsamples-nvalid_samples
            
        self.train_data = data[:ntrain_samples]

        if is_test:
            self.valid_data = None
        else:
            self.valid_data = data[ntrain_samples:]
        
        if max_gpu_train_samples > ntrain_samples:
            max_gpu_train_samples = ntrain_samples
        
        if max_gpu_valid_samples > nvalid_samples:
            max_gpu_valid_samples = nvalid_samples
        
        self.shared_train_data = theano.shared(numpy.zeros((max_gpu_train_samples, 
                                                            data.shape[1]), 
                                                           dtype=data.dtype),
                                                       borrow=True)
        
        if not is_test:
            self.shared_valid_data =  theano.shared(numpy.zeros((max_gpu_valid_samples, 
                                                                    data.shape[1]), 
                                                                   dtype=data.dtype),
                                                            borrow=True)

            
        self.ntrain_data = self.train_data.shape[0]
        self.nvalid_data = nvalid_samples
        # set the first chunck of data on shared variables
        self.shared_train_data.set_value(self.train_data[:max_gpu_train_samples])
        
        if not is_test and self.shared_valid_data:
            self.shared_valid_data.set_value(self.valid_data[:max_gpu_valid_samples])
        
        self.max_gpu_train_samples = max_gpu_train_samples
        self.max_gpu_valid_samples = max_gpu_valid_samples
        # set gpu batch and chuck index to the begninning
        self.gpu_train_chunk_index = self.gpu_valid_chunk_index = 0
        self.ntrain_batches_gpu = int(ceil(max_gpu_train_samples/float(batch_size)))
        self.nvalid_batches_gpu = int(ceil(max_gpu_valid_samples/float(batch_size)))
    
    def get_data_stats(self):
        return numpy.mean(self.train_data, axis=0)
    
    def get_number_of_train_batches(self):
        return int(ceil(self.ntrain_data/float(self.batch_size)))

    def get_number_of_valid_batches(self):
        return int(ceil(self.nvalid_data/float(self.batch_size)))
    
    def get_number_of_train_data(self):
        return self.ntrain_data
    
    def get_number_of_valid_data(self):
        return self.nvalid_data
    
    def get_train_data_and_idx(self, minibatch_index):
        assert minibatch_index < ceil(float(self.ntrain_data)/self.batch_size), \
            'using minibatch of index %d (starting from 0) but only have %d minibatches' % (minibatch_index, ceil(float(self.ntrain_data)/self.batch_size))
        # perturbe the data every epoch
        if not self.is_test and self.epochwise_shuffle and minibatch_index == 0:
            perm_idx = numpy.random.permutation(self.ntrain_data)
            if self.perturb_function is not None:
                train_data = self.perturb_function(self.train_data[perm_idx])
            else:
                train_data = self.train_data[perm_idx]
        else:
            if self.perturb_function is not None:
                train_data = self.perturb_function(self.train_data)
            else:
                train_data = self.train_data
            
        gpu_sample_start_index, gpu_sample_end_index, self.gpu_train_chunk_index = \
                UnlabeledMemoryDataProvider.__update_gpu_batch_data(minibatch_index=minibatch_index, 
                                                       nbatches_per_gpu=self.ntrain_batches_gpu,
                                                       batch_size=self.batch_size,
                                                       gpu_chunck_index=self.gpu_train_chunk_index, 
                                                       max_gpu_samples=self.max_gpu_train_samples, 
                                                       total_num_samples=self.ntrain_data,
                                                       data=train_data, 
                                                       out_gpu_data=self.shared_train_data)
                
        return self.shared_train_data, gpu_sample_start_index, gpu_sample_end_index

    def get_valid_data_and_idx(self, minibatch_index):
        assert not self.is_test, ('no validation data for test data')
        assert minibatch_index < ceil(float(self.ntrain_data)/self.batch_size), \
            'using minibatch of index %d (starting from 0) but only have %d minibatches' % (minibatch_index, self.nvalid_data//self.batch_size)
        gpu_sample_start_index, gpu_sample_end_index, self.gpu_valid_chunk_index = \
                UnlabeledMemoryDataProvider.__update_gpu_batch_data(minibatch_index=minibatch_index, 
                                                       nbatches_per_gpu=self.nvalid_batches_gpu, 
                                                       batch_size=self.batch_size,
                                                       gpu_chunck_index=self.gpu_valid_chunk_index, 
                                                       max_gpu_samples=self.max_gpu_valid_samples, 
                                                       total_num_samples=self.nvalid_data,
                                                       data=self.valid_data, 
                                                       out_gpu_data=self.shared_valid_data)
                
        return self.shared_valid_data, gpu_sample_start_index, gpu_sample_end_index
    
    @staticmethod
    def __update_gpu_batch_data(minibatch_index, nbatches_per_gpu, batch_size,
                                gpu_chunck_index, max_gpu_samples,
                                data, total_num_samples,
                                out_gpu_data):
        '''update the large chunck of data on gpu if necessary, based on
           the minibatch_index, and the gpu_chunck_index'''
        gpu_batch_index = minibatch_index % nbatches_per_gpu
        if minibatch_index // nbatches_per_gpu != gpu_chunck_index:
            gpu_chunck_index = minibatch_index // nbatches_per_gpu
            
            start_idx = gpu_chunck_index*max_gpu_samples
            end_idx = min(total_num_samples, (gpu_chunck_index+1)*max_gpu_samples)

            out_gpu_data.set_value(data[start_idx:end_idx], borrow=True)
        
        if (minibatch_index == ceil(total_num_samples/float(batch_size))-1 
            and ceil(total_num_samples/float(batch_size)) != total_num_samples//batch_size): 
            # the last batch
            # rewind backwards a bit
            if total_num_samples%max_gpu_samples != 0:
                sample_start_index = (total_num_samples%max_gpu_samples)-batch_size
                sample_end_index = (total_num_samples%max_gpu_samples)
            else:
                sample_start_index = total_num_samples-batch_size
                sample_end_index = total_num_samples
        else:
            sample_start_index = gpu_batch_index*batch_size
            sample_end_index = (gpu_batch_index+1)*batch_size
        return sample_start_index, sample_end_index, gpu_chunck_index
    
