import torch
import numpy as np
import random

class QDRLTokenWindow:

    def __init__(
            self,
            batch,
            bd_dims:int,
            obs_dims:int,
            act_dims:int):
        """
        batch should be of shape batch_size*M, with M=ep_len*(bd_dims+obs_dims+act_dims)

        each example batch[ex,:] is a 1D tensor that semantically can be separated into
                       [ bd_0, obs_0, act_0,
                         bd_1, obs_1, act_1,
                         ...
                         bd_{N-1], obs_{N-1}, act_{N-1}]

        with bd_i, obs_i, act_i respectively of lengths bd_dims, obs_dims and act_dims.
        """
        self.batch=batch
        self.bd_dims=bd_dims
        self.obs_dims=obs_dims
        self.act_dims=act_dims

        self._M=self.batch.shape[1]
        self._D=self.bd_dims+self.obs_dims+self.act_dims
        self._N=self._M//(self._D)#episode length
        self._L=3*self._N#this is the total number of tokens in the batch
       
        #compute indexes for bds. Those would be at [0:bd_dims], [D:D+bd_dims], ..., [(N-1)D:(N-1)D+bd_dims]
        self.bd_indexes=[]
        for l_i in range(self._N):
            start_index=self._D*l_i
            end_index=start_index+self.bd_dims
            self.bd_indexes+=list(np.arange(start_index, end_index))

        #compute indexes for obs. Those would be at [bd_dims:bd_dims+obs_dims], [D+bd_dims:D+bd_dims+obs_dims], ..., [(N-1)D+bd_dims:(N-1)D+bd_dims+obs_dims]
        self.obs_indexes=[]
        for l_i in range(self._N):
            start_index=self._D*l_i+self.bd_dims
            end_index=start_index+self.obs_dims
            self.obs_indexes+=list(np.arange(start_index, end_index))

        #compute indexes for act. Those would be at [bd_dims+obs_dims:bd_dims+obs_dims+act_dims], [D+bd_dims+obs_dims:D+bd_dims+obs_dims+act_dims], ..., etc
        self.act_indexes=[]
        for l_i in range(self._N):
            start_index=self._D*l_i+self.bd_dims+self.obs_dims
            end_index=start_index+self.act_dims
            self.act_indexes+=list(np.arange(start_index, end_index))

    @property
    def episode_len(self):
        return self._N

    def get_bd_tensor(self):
        """
        Note that this returns a copy, not a view
        """
        return self.batch[:,self.bd_indexes]

    def get_obs_tensor(self):
        """
        Note that this returns a copy, not a view
        """
        return self.batch[:,self.obs_indexes]

    def get_act_tensor(self):
        """
        Note that this returns a copy, not a view
        """
        return self.batch[:,self.act_indexes]



if __name__=="__main__":

    _seed=0
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    
    _batch_sz=2 
    _ep_len=10
    _bd_dims=2
    _obs_dims=5
    _act_dims=3
    _bb=torch.floor(torch.rand(_batch_sz, _ep_len*(_bd_dims+_obs_dims+_act_dims))*100)
    
    manager=QDRLTokenWindow(
            _bb,
            bd_dims=_bd_dims,
            obs_dims=_obs_dims,
            act_dims=_act_dims)
