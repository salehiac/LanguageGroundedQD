import torch
import numpy as np
import random

class QDRLTokenWindow:

    def __init__(
            self,
            batch,
            bd_dims:int,
            obs_dims:int,
            act_dims:int,
            timestep:int):
        """

        Args: 
            batch (torch.Tensor): should be of shape batch_size*M, with M=ep_len*(bd_dims+obs_dims+act_dims)
                                  each example batch[ex,:] is a 1D tensor that semantically can be separated into
                                  [ bd_0, obs_0, act_0,
                                    bd_1, obs_1, act_1,
                                    ...
                                    bd_{N-1], obs_{N-1}, act_{N-1}]

                                  with bd_i, obs_i, act_i respectively of lengths bd_dims, obs_dims and act_dims.
            bd_dims (int): length of the behavior descriptor
            obs_dims (int): length of observation vector 
            act_dims (int): lenght of action vector
            timestep (int): timestep of obs_0 (since the sequences from the batch could be subtrajectories from full episodes).

        """
        self.batch=batch
        self._B=batch.shape[0]
        self.bd_dims=bd_dims
        self.obs_dims=obs_dims
        self.act_dims=act_dims
        self.timestep=timestep

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
        return self.batch[:,self.bd_indexes].view(self._B, self._N, self.bd_dims)

    def get_obs_tensor(self):
        return self.batch[:,self.obs_indexes].view(self._B, self._N, self.obs_dims)

    def get_act_tensor(self):
        return self.batch[:,self.act_indexes].view(self._B, self._N, self.act_dims)

    def get_position_tensor(self):
        """
        Input to postional embedding layer. The shape will be B*N
        """
        return torch.arange(self._N).repeat(self._B,1)+self.timestep

    def embedding_to_sequence(self,bds_emb, obs_emb, acts_emb):
        """
        takes the outputs of the embedding layers for behavior descriptors, observations and actions and rearranges them into sequences as in the original episode

        each of the input tensors should be of shape B*N*H, with H the embedding_dims
        """
        mat=torch.cat([bds_emb, obs_emb, acts_emb],-1)
        ret_mat=mat.reshape(self._B, -1)
        return ret_mat


class RLTokenEmbeddingMLP(torch.nn.Module):

    def __init__(self, in_sz, emb_sz, dropout, bias=True):
        
        super().__init__()
        self.l1=torch.nn.Linear(in_sz, emb_sz, bias)
        self.nonlin=torch.nn.GELU()
        self.l2=torch.nn.Linear(emb_sz, emb_sz, bias)
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self, x):
        x=self.l1(x)
        x=self.nonlin(x)
        x=self.l2(x)
        x=self.dropout(x)
        return x

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
   
    _ts=5
    manager=QDRLTokenWindow(
            _bb,
            bd_dims=_bd_dims,
            obs_dims=_obs_dims,
            act_dims=_act_dims,
            timestep=_ts)

    test_rearrange_embedding=True 
    if test_rearrange_embedding:

        _cc=manager.embedding_to_sequence(manager.get_bd_tensor(), manager.get_obs_tensor(), manager.get_act_tensor())
        assert (_cc==_bb).all(), "in this test, the reconstructed _cc should be the same as the original batch _bb"
        print("test passed.")
    
    test_token_embeddings=True
    if test_token_embeddings:
        
        _emb_sz=64
        _mlp_bds=RLTokenEmbeddingMLP(_bd_dims, _emb_sz, dropout=0.1)
        _mlp_obs=RLTokenEmbeddingMLP(_obs_dims, _emb_sz, dropout=0.1)
        _mlp_acts=RLTokenEmbeddingMLP(_act_dims, _emb_sz, dropout=0.1)

        _emb_bds=_mlp_bds(manager.get_bd_tensor())
        _emb_obs=_mlp_obs(manager.get_obs_tensor())
        _emb_acts=_mlp_acts(manager.get_act_tensor())

        _dd=manager.embedding_to_sequence(_emb_bds, _emb_obs, _emb_acts)
