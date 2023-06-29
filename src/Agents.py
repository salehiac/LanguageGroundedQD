from abc import ABC, abstractmethod
import time
import copy
import pdb

import torch
import numpy as np
import random
from scoop import futures

class Agent(ABC):
    """
    Baseclass for reactive policies stored in the Archive
    TransformerBased Policies can be found in nanoGPT_QDRL/model.py
    """
    @abstractmethod
    def get_flattened_weights(self):
        pass
    @abstractmethod
    def set_flattened_weights(self, w):
        pass
    
    @abstractmethod
    def get_genotype_len(self):
        pass
    
    
    def __init__(self, idx):
        self._fitness=None
        self._tau=None#a trajectory {(o_i, a_i)}_i
        self._behavior=None#behavior is the complete traj in behavior space, which is mapped to some _behavior_descr
        self._behavior_descr=None
        self._annotation=None
        self._llm_descr=None
        self._nov=None
        self._idx=idx

        self._solved_task=False
        self._task_info={}
        self._created_at_gen=-1 
        self._parent_idx=-1
        self._root=-1#useful for meta-QD problems
        self._bd_dist_to_parent_bd=-1
        self._age=-1

        #only useful for meta-learning with MetaQDForSparseRewards
        self._useful_evolvability=0
        self._mean_adaptation_speed=float("inf")
        self._adaptation_speed_lst=[]


    def reset_tracking_attrs(self):
        self._fitness=None
        self._behavior_descr=None
        self._nov=None

        self._solved_task=False
        self._task_info={}
        self._created_at_gen=-1 
        self._parent_idx=-1
        self._root=-1
        self._bd_dist_to_parent_bd=-1
        self._age=-1

        self._useful_evolvability=0
        self._mean_adaptation_speed=float("inf")
        self._adaptation_speed_lst=[]

    def to_batch_example(self):
        """
        returns [textual_conditioning, step_by_step_inputs] with the second element a 1d tensor of shape (episode_len*M) with M=bd_dims+action_dims+obs_dims. 
        """
        text=copy.deepcopy(self._llm_descr)
        
        actions=torch.tensor(self._tau["action"])
        obs=torch.tensor(self._tau["obs"])
        N=obs.shape[0]#episode length
        bd_conditioning=torch.tensor(self._behavior_descr).repeat(N,1)

        step_by_step_inputs=torch.cat([bd_conditioning, obs, actions],1)

        return (text, step_by_step_inputs.reshape(N*(obs.shape[1]+actions.shape[1]+bd_conditioning.shape[1])))#reshapes in a row major manner


_non_lin_dict={"tanh":torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}

def get_num_number_params(model, trainable_only=False):
    if trainable_only:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
    else:
        model_parameters = model.parameters()

    n_p = sum([np.prod(p.size()) for p in model_parameters])

    return n_p

def get_params_sum(model, trainable_only=False):

    with torch.no_grad():
        if trainable_only:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
        else:
            model_parameters = model.parameters()

        u=sum([x.sum().item() for x in model_parameters])
        return u

def _identity(x):
    """
    because pickle and thus scoop don't like lambdas...
    """
    return x

class SmallFC_FW(torch.nn.Module, Agent):

    def __init__(self,
            idx,
            in_d,
            out_d,
            num_hidden=3,
            hidden_dim=10,
            non_lin="tanh",
            output_normalisation=""):
        torch.nn.Module.__init__(self)
        Agent.__init__(self, idx)

        self.mds=torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])
        
        for i in range(num_hidden-1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))


        self.non_lin=_non_lin_dict[non_lin] 

        self.output_normaliser=_non_lin_dict[output_normalisation] if output_normalisation else _identity


    def forward(self, x, return_numpy=True):
        """
        x list
        """
        out=torch.Tensor(x).unsqueeze(0)
        for md in self.mds[:-1]:
            out=self.non_lin(md(out))
        out=self.mds[-1](out)
        out=self.output_normaliser(out)
        return out.detach().cpu().numpy() if return_numpy else out

    def get_flattened_weights(self):
        """
        returns list 
        """
        flattened=[]
        for m in self.mds:
            flattened+=m.weight.view(-1).tolist()
            flattened+=m.bias.view(-1).tolist()

        return flattened
    
    def set_flattened_weights(self, w_in):
        """
        w_in list
        """

        with torch.no_grad():
            assert len(w_in)==get_num_number_params(self, trainable_only=True), "wrong number of params"
            start=0
            for m in self.mds:
                w=m.weight
                b=m.bias
                num_w=np.prod(list(w.shape))
                num_b=np.prod(list(b.shape))
                m.weight.data=torch.Tensor(w_in[start:start+num_w]).reshape(w.shape)
                m.bias.data=torch.Tensor(w_in[start+num_w:start+num_w+num_b]).reshape(b.shape)
                start=start+num_w+num_b

    def zero_out(self):
        """
        debug function
        """
        with torch.no_grad():
            for m in self.mds:
                m.weight.fill_(0.0)
                m.bias.fill_(0.0)
       
    def check_set_get_flattened_weights(self):
        res=[get_params_sum(self)]
       
        z=self.get_flattened_weights()
        
        self.zero_out()
        res.append(get_params_sum(self))

        self.set_flattened_weights(z)
        res.append(get_params_sum(self))

        test_passed= (res[0]==res[2] and res[1]==0)

        assert test_passed, "this shouldn't happen"
        return test_passed

    def get_genotype_len(self):
        return get_num_number_params(self, trainable_only=True)

