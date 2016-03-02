from data_provider import UnlabeledDataProvider, DataProvider
from math import ceil

import h5py
import numpy
import theano
import warnings
from data.data_provider import UnlabeledDataProvider

__all__ = ['DiskDataProvider', 'UnlabeledDiskDataProvider']

class DiskDataProvider(DataProvider):
    
    '''
       DiskDataProvider assumes the full data is on disk,
       it is best suited for datasets that cannot fit in memory.
    '''
    
    def __init__(self, data_loader,
                 batch_size = 100,
                 max_ram_train_samples=100000,
                 max_ram_valid_samples=100000,
                 max_gpu_train_samples=10000,
                 max_gpu_valid_samples=10000,
                 nvalid_samples=0,
                 is_test=False,
                 epochwise_shuffle=False,
                 perturb_function=None,
                 seed=123):
        
        assert max_ram_train_samples%max_gpu_train_samples==0, 'max_ram_train_samples need to be multiple of max_gpu_train_samples'
        assert max_ram_valid_samples%max_gpu_valid_samples==0, 'max_ram_valid_samples need to be multiple of max_gpu_valid_samples'
        assert max_gpu_train_samples%batch_size==0, 'max_gpu_train_samples need to be multiple of batch_size'
        assert max_gpu_valid_samples%batch_size==0, 'max_gpu_valid_samples need to be multiple of batch_size'
        
        super(DiskDataProvider, self).__init__(data_loader=data_loader, 
                                             batch_size=batch_size,
                                             nvalid_samples=nvalid_samples,
                                             is_test=is_test)
        
        self.rng = numpy.random.RandomState(seed)
        self.epochwise_shuffle = epochwise_shuffle
        self.perturb_function = perturb_function
        self.data_loader = data_loader
        
        if not is_test and nvalid_samples == 0 and data_loader.get_nvalid_data()<=0:
            warnings.warn('No validation samples provided')
        
        # in case there is no validation data from the data loader
        if data_loader.get_nvalid_data() <= 0:
            self.ntrain_data = data_loader.get_ntrain_data() - nvalid_samples
            self.nvalid_data = nvalid_samples
        else:
            self.ntrain_data = data_loader.get_ntrain_data()
            self.nvalid_data = data_loader.get_nvalid_data()
        
    def get_number_of_train_batches(self):
        return int(ceil(self.ntrain_data/float(self.batch_size)))

    def get_number_of_valid_batches(self):
        return int(ceil(self.nvalid_data/float(self.batch_size)))
        
    def get_number_of_train_data(self):
        return self.ntrain_data
    
    def get_number_of_valid_data(self):
        return self.nvalid_data
    
    @staticmethod
    def _update_ram_batch_data(minibatch_index, nbatches_per_ram, 
                                ram_chunk_index, 
                                max_ram_samples, total_num_samples,
                                data, label,
                                out_ram_data, out_ram_label, shuffle_data=False,
                                perturb_function=None):
        '''update the large chunck of data on ram if necessary, based on
           the minibatch_index, and the ram_chunck_index'''
        ram_batch_index = minibatch_index % nbatches_per_ram
        start_idx = ram_chunk_index*max_ram_samples
        end_idx = min(total_num_samples, (ram_chunk_index+1)*max_ram_samples)
        ram_updated = False
        if minibatch_index // nbatches_per_ram != ram_chunk_index:
            ram_chunk_index = minibatch_index // nbatches_per_ram
            start_idx = ram_chunk_index*max_ram_samples
            end_idx = min(total_num_samples, (ram_chunk_index+1)*max_ram_samples)
            ram_updated = True
            if perturb_function is not None:
                if out_ram_label is not None:
                    out_ram_data[:end_idx-start_idx], out_ram_label[:end_idx-start_idx] = \
                        perturb_function(data[start_idx:end_idx], label[start_idx:end_idx])
                else:
                    out_ram_data[:end_idx-start_idx] = \
                        perturb_function(data[start_idx:end_idx])
            else:
                out_ram_data[:end_idx-start_idx] = data[start_idx:end_idx]
                
                if out_ram_label is not None:
                    out_ram_label[:end_idx-start_idx] = label[start_idx:end_idx]
            
            if shuffle_data:
                perm_idx = numpy.random.permutation(end_idx-start_idx)
                out_ram_data[:end_idx-start_idx] = out_ram_data[perm_idx]
                if out_ram_label is not None:
                    out_ram_label[:end_idx-start_idx] = out_ram_label[perm_idx]
                    
        return ram_batch_index, ram_chunk_index, end_idx-start_idx, ram_updated
    
    @staticmethod
    def _update_gpu_batch_data(minibatch_index, nbatches_per_gpu, ram_updated,
                                gpu_chunck_index, max_gpu_samples,
                                data, label, current_ram_samples,
                                out_gpu_data, out_gpu_label):
        '''update the large chunck of data on gpu if necessary, based on
           the minibatch_index, and the gpu_chunck_index'''
        gpu_batch_index = minibatch_index % nbatches_per_gpu
        if minibatch_index // nbatches_per_gpu != gpu_chunck_index or ram_updated:
            gpu_chunck_index = minibatch_index // nbatches_per_gpu
            
            start_idx = gpu_chunck_index*max_gpu_samples
            end_idx = min(current_ram_samples, (gpu_chunck_index+1)*max_gpu_samples)

            out_gpu_data.set_value(data[start_idx:end_idx], borrow=True)
            
            if out_gpu_label is not None:
                out_gpu_label.set_value(label[start_idx:end_idx], borrow=True)
            
        if (minibatch_index == ceil(current_ram_samples/float(batch_size))-1 
            and ceil(current_ram_samples/float(batch_size)) != current_ram_samples//batch_size): 
            # the last batch
            # rewind backwards a bit
            if current_ram_samples%max_gpu_samples != 0:
                sample_start_index = (current_ram_samples%max_gpu_samples)-batch_size
                sample_end_index = (current_ram_samples%max_gpu_samples)
            else:
                sample_start_index = current_ram_samples-batch_size
                sample_end_index = current_ram_samples
        else:
            sample_start_index = gpu_batch_index*batch_size
            sample_end_index = (gpu_batch_index+1)*batch_size
            
        return sample_start_index, sample_end_index, gpu_chunck_index
    
class UnlabeledDiskDataProvider(DiskDataProvider, UnlabeledDataProvider):
    
    def __init__(self,
                 data_loader,
                 batch_size = 100,
                 max_ram_train_samples=100000,
                 max_ram_valid_samples=100000,
                 max_gpu_train_samples=10000,
                 max_gpu_valid_samples=10000,
                 nvalid_samples=0,
                 is_test=False,
                 epochwise_shuffle=False,
                 perturb_function=None,
                 seed=123):
        
        super(UnlabeledDiskDataProvider, self).__init__(data_loader=data_loader,
                 batch_size=batch_size,
                 max_ram_train_samples=max_ram_train_samples,
                 max_ram_valid_samples=max_ram_valid_samples,
                 max_gpu_train_samples=max_gpu_train_samples,
                 max_gpu_valid_samples=max_gpu_valid_samples,
                 nvalid_samples=nvalid_samples,
                 is_test=is_test,
                 epochwise_shuffle=epochwise_shuffle,
                 perturb_function=perturb_function,
                 seed=seed)
        
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
            
        self.train_data = data
        
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
        
        self.shared_valid_data = None
        if not is_test and nvalid_samples > 0:
            
            self.shared_valid_data =  theano.shared(numpy.zeros((max_gpu_valid_samples, 
                                                                    data.shape[1]), 
                                                                   dtype=data.dtype),
                                                            borrow=True)

        self.max_ram_train_samples = min(max_ram_train_samples, self.ntrain_data)
        self.max_ram_valid_samples = min(max_ram_valid_samples, self.nvalid_data)
        self.max_gpu_train_samples = max_gpu_train_samples
        self.max_gpu_valid_samples = max_gpu_valid_samples
        
        # the last ram chunck need special care in case the total number of samples
        # does not divide max ram samples
        self.is_train_ram_chunck_divide = (self.ntrain_data%self.max_ram_train_samples==0)
        if self.max_ram_valid_samples > 0:
            self.is_valid_ram_chunck_divide = (self.nvalid_data%self.max_ram_valid_samples==0)
        else:
            self.is_valid_ram_chunck_divide = True
        
        
        # get the disk chunck of data to ram
        self.ram_train_data = self.train_data[:max_ram_train_samples]
        if self.valid_data:
            self.ram_valid_data = self.valid_data[:max_ram_valid_samples]
        else:
            self.ram_valid_data = None 
        
        # set the first chunck of data on shared variables
        # self.train_idx take care of the shuffle data case
        self.shared_train_data.set_value(self.ram_train_data[:max_gpu_train_samples])
        
        if self.shared_valid_data:
            # no need to shuffle validation data
            self.shared_valid_data.set_value(self.ram_valid_data[:max_gpu_valid_samples])

        # set gpu batch and chuck index to the begninning
        self.ram_train_chunk_index = self.ram_valid_chunk_index = 0
        self.gpu_train_chunk_index = self.gpu_valid_chunk_index = 0
        
        # supply all data to gpu
        self.ntrain_batches_ram = self.max_ram_train_samples//self.batch_size
        self.nvalid_batches_ram = self.max_ram_valid_samples//self.batch_size
        self.ntrain_batches_gpu = self.max_gpu_train_samples//self.batch_size
        self.nvalid_batches_gpu = self.max_gpu_valid_samples//self.batch_size
    

    def get_train_data_and_idx(self, minibatch_index):

        # update ram data
        ram_batch_index, self.ram_train_chunk_index, crt_ram_samples, ram_updated = \
             UnlabeledDiskDataProvider._update_ram_batch_data(minibatch_index=minibatch_index, 
                                                          nbatches_per_ram=self.ntrain_batches_ram,
                                                          ram_chunk_index=self.ram_train_chunk_index, 
                                                          max_ram_samples=self.max_ram_train_samples, 
                                                          total_num_samples=self.ntrain_data, 
                                                          data=self.train_data, 
                                                          label=None, 
                                                          out_ram_data=self.ram_train_data, 
                                                          out_ram_label=None,
                                                          perturb_function=self.perturb_function,
                                                          shuffle_data=self.epochwise_shuffle)
        # update gpu data
        gpu_sample_start_index, gpu_sample_end_index, self.gpu_train_chunk_index = \
                UnlabeledDiskDataProvider._update_gpu_batch_data(minibatch_index=ram_batch_index, 
                                                       nbatches_per_gpu=self.ntrain_batches_gpu, 
                                                       gpu_chunck_index=self.gpu_train_chunk_index, 
                                                       max_gpu_samples=self.max_gpu_train_samples, 
                                                       current_ram_samples=crt_ram_samples,
                                                       data=self.ram_train_data, 
                                                       label=None, 
                                                       out_gpu_data=self.shared_train_data, 
                                                       out_gpu_label=None,
                                                       ram_updated = ram_updated)
                
        return self.shared_train_data, gpu_sample_start_index, gpu_sample_end_index

    def get_valid_data_and_idx(self, minibatch_index):
        # update ram data
        ram_batch_index, self.ram_valid_chunk_index, crt_ram_samples, ram_updated = \
             UnlabeledDiskDataProvider._update_ram_batch_data(minibatch_index=minibatch_index, 
                                                          nbatches_per_ram=self.nvalid_batches_ram,
                                                          ram_chunk_index=self.ram_valid_chunk_index, 
                                                          max_ram_samples=self.max_ram_valid_samples, 
                                                          total_num_samples=self.nvalid_data, 
                                                          data=self.valid_data, 
                                                          label=None, 
                                                          out_ram_data=self.ram_valid_data, 
                                                          out_ram_label=None)
        # update gpu data
        gpu_sample_start_index, gpu_sample_end_index, self.gpu_valid_chunk_index = \
                UnlabeledDiskDataProvider._update_gpu_batch_data(minibatch_index=ram_batch_index, 
                                                       nbatches_per_gpu=self.nvalid_batches_gpu, 
                                                       gpu_chunck_index=self.gpu_valid_chunk_index, 
                                                       max_gpu_samples=self.max_gpu_valid_samples, 
                                                       current_ram_samples=crt_ram_samples,
                                                       data=self.ram_valid_data, 
                                                       label=None, 
                                                       out_gpu_data=self.shared_valid_data, 
                                                       out_gpu_label=None,
                                                       ram_updated=ram_updated)
                
        return self.shared_valid_data, gpu_sample_start_index, gpu_sample_end_index
    
    def get_data_stats(self):
        return self.data_loader.get_data_stats()
    
# test
if __name__ == '__main__':

    from lsun_data_loader import UnlabeledLSUNDataLoader
    numpy.random.seed(123)
    test_train_data = numpy.asarray(numpy.random.randint(255, size=(1281167,10)), dtype='uint8')
    print test_train_data[0,:]
    with h5py.File('test_train_tmp.hdf5', 'w') as f:
        d = f.create_dataset('data', data=test_train_data)
        d = test_train_data
    
    batch_size = 250
    max_ram_train_samples = 300*250
    max_ram_valid_samples = 100*250
    max_gpu_train_samples = 100*250
    max_gpu_valid_samples = 10*250
    p = UnlabeledDiskDataProvider(data_loader=UnlabeledLSUNDataLoader('test_train_tmp.hdf5'), 
                                  batch_size=batch_size, 
                                  max_ram_train_samples=max_ram_train_samples, 
                                  max_ram_valid_samples=max_ram_valid_samples, 
                                  max_gpu_train_samples=max_gpu_train_samples, 
                                  max_gpu_valid_samples=max_gpu_valid_samples, 
                                  nvalid_samples=0)
    
    print p.get_number_of_train_batches()
    if int(ceil(test_train_data.shape[0]/float(batch_size))) == p.get_number_of_train_batches():
        print 'passed train batches'
    else:
        print 'failed train batches'
    
    failed = False
    for i in xrange(2):
        for minibatch_idx in xrange(test_train_data.shape[0]//batch_size):
            s_d, gpu_start_idx, gpu_end_idx = p.get_train_data_and_idx(minibatch_idx)
            if (numpy.all(numpy.equal(s_d.get_value()[gpu_start_idx:gpu_end_idx],
                                                  test_train_data[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size]))):
                continue
            else:
                failed = True
                print 'minibatch %d failed at epoch %d' % (minibatch_idx, i)
                break
    if not failed:
        print 'passed all data tests'
     
