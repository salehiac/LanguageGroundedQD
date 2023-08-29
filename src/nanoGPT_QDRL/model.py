"""
This code extends nanoGPT (https://github.com/karpathy/nanoGPT) for conditioning on language and behavior descriptors
"""

import math
import inspect
from collections import namedtuple
from typing import Literal, List
import pdb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from termcolor import colored

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))  #### It is very important for it to be torch.tril, if it's upper triangular then QK^TV wont work
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))#causal mask
            assert T==self.bias.shape[-1]==self.bias.shape[-2], "this should not happend."#this assert is used because of the previous line: why did the original code use :T, instead of ,:]?
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def _identity(x):
    """
    because pickle doesn't like lambdas
    """
    return x

class RLMLP(torch.nn.Module):
    """
    the main reason this is separate from the MLP class here is that the MLP module is used in residual layers  
    and its c_fc is initialized differently because of that. I didn't want to change that behavior from the original repo
    """

    def __init__(self, in_sz, h_sz, out_sz, dropout=0.0, bias=True, scale_out_put=-1):
        
        super().__init__()
        self.l1=torch.nn.Linear(in_sz, 2*h_sz, bias)
        self.l2=torch.nn.Linear(2*h_sz, 2*h_sz, bias)
        self.l3=torch.nn.Linear(2*h_sz, out_sz, bias)
        self.nonlin=torch.nn.GELU()
        self.dropout=torch.nn.Dropout(dropout) if dropout else _identity
        self.scale_out_put=scale_out_put

    def forward(self, x):
        x=self.l1(x)
        x=self.nonlin(x)
        x=self.l2(x)
        x=self.nonlin(x)
        x=self.l3(x)
        x=self.dropout(x)
       
        if self.scale_out_put!=-1:
            x=torch.tanh(x)*self.scale_out_put

        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

GPT_QDRLConfig=namedtuple("GPT_QDRLConfig",[
    "block_size", #this is the blocksize in tokens, not characters. It is used to specify the positional embeddings as torch.nn.Embedding(block_size, n_embd)
                  #contrary to the original transformer paper, those embeddings are learned too here.
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",# True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    "n_action_dims",#added for QD-RL
    "n_obs_dims",#added for QD-RL
    "n_bd_dims",#added for QD-RL
    "kmeans_obj_lst",
    ])

   
def process_batch(
        batch,
        tokenizer,
        context_size,
        bd_dims,
        obs_dims,
        act_dims,
        device,
        kmeans_lst,
        input_normalizer=None,
        max_len_pad=226):
    """
    Args: 
        batch (list): a list of length 3, with
                        - batch[0] a list of batch_size strings
                        - batch[1] a tensor of shape batch_size*M, with M=ep_len*(bd_dims+obs_dims+act_dims). Note that actions in this tensor are offsets relative to a cluster center
                          each example batch[ex,:] is a 1D tensor that semantically can be separated into
                                        [ bd_0, obs_0, act_0,
                                          bd_1, obs_1, act_1,
                                          ...
                                          bd_{N-1], obs_{N-1}, act_{N-1}] #with N the episode length. Note that all example trajectories are assumed to have the same length
                                                                          #Note that each of the bd_i, obs_i, act_i are considered as a separate token, so 
                                                                          #there are a total of 3N tokens in each trajectory

                                        with bd_i, obs_i, act_i respectively of lengths bd_dims, obs_dims and act_dims.
                        - batch[2] a tensor of shape batch_size*ep_len*1, which gives the cluster center idx for each action.
        
        tokenizer (PreTrainedTokenizerFast): A tokenizer pretrained on the corpus
        context_size (int): the context size of the transformer
        bd_dims (int): length of behavior descriptors
        obs_dims (int): length of observations vector
        act_dims (int): length of action vector
        kmeans_lst (List[Mkeans]): list of objects encapsulating cluster center ids and their coordinates

    Returns:
        (See the notes section below for the defintion of T_u.)

        text_token_ids (torch.LongTensor): shape batch_size*T_text, with T_text the number of tokkens after padding to the token length of the longest string in the batch.
        text_posional_ids (torch.LongTensor): 1d tensor of shape T_text, to compute positional embeddings.
        bd_tensor (torch.tensor): shape batch_size*T_u*bd_dims. 
        obs_tensor (torch.tensor): shape batch_size*T_u*obs_dims. 
        act_tensor (torch.tensor): shape batch_size*T_u*act_dims. Those actions are the same as cluster_coord+offset.
        act_tensor_cluster_id (torch.tensor): shape batch_size*T_u. Gives the cluster id that corresponds to an action. Useful for supervision.
        act_tensor_cluster_center_offsets (torch.tensor): shape batch_size*T_u. Gives the offset between the action and the cluster center. Useful for supervision without going through the kmeans
                                                          object again
        subseq_timestamps (torch.LongTensor): 1d tensor of shape T_u. See the notes section below for the defintion of T_u.
    Notes:
        - The context length of the attention blocks is smaller than the full number of tokens (3N) in the trajectory. Let's note
            T_u=context_size-T_text 
          the number of tokens that must be selected. Such a sequence of consecutive tokens should always start on a bd_j.

          Let us note F=3*floor(T_u/3)) and R=T_u%3. For simplicity, we select a subsequence of F tokens

            bd_j, obs_j, act_j, ..., bd_{T_u//3}, obs_{T_u//3}, act_{T_u//3} 

          and if R!=0, we'll just use padding after computing the embeddings (see the next note below), just before feeding the context to the first attention block. Note that in that
          case, including the real tokens wouldn't be of any use anyway because they wouldn't intervene in the loss (as the aim is to predict the actions) nor in the attention (as it is
          causal). The indexes j for which a selection of this length is possible are {j|j<=N-{T_u//3}}, and one of them is selected randomly.

          This gives us bd, obs and act tensors of shapes batch_size*{T_u//3}*bd_dims.

        - On padding: While padding for text happends in this preprocessing, padding for the QD-RL trajectory happends later. During the forward pass, those tensors will be passed
          to three MLPs each providing embeddings for the bds, obs and actions, and then those embeddings are rearranged as in their original order in the sequence. This will result
          in and embedding of shape batch_size*F*embedding_dims. At this point, we will add paddings of appropriate dimensions if necessary. 
    """

    text_batch=batch[0]
    text_token_ids=tokenizer(text_batch, padding="max_length", max_length=max_len_pad, return_tensors="pt").input_ids #padding might not be the most optimal way, but it simplifies things
    T_text=num_text_tokens=text_token_ids.shape[1]
    text_posional_ids=torch.arange(T_text,dtype=torch.long)
 
    T_u=context_size-T_text
    #print(f"T_u=={T_u}")

    min_RL_timestamp=10#this is arbitrary and just for the assert
    assert T_u//3>min_RL_timestamp, f"The text from the batch leave room for less than {T_u} RL tokens. Either your text is too long, or you should increase the context size (config.block_size)"

    BB=batch[1].shape[0]
    DD=bd_dims+obs_dims+act_dims
    NN=batch[1].shape[1]//DD #episode length
    traj_batch=batch[1].reshape(BB,NN,DD).float() #traj_batch[ex_i,j,:] is bd_j, obs_j, act_j

    assert NN-T_u//3+1<=0, f"context lenght should be long enough to include the full trajectory (NN={NN}, T_u={T_u}, block_size={context_size}, T_text={T_text}"
    possible_js=torch.arange(0,max(1,NN-T_u//3+1),dtype=torch.long)
    jj=torch.multinomial(torch.ones_like(possible_js).float()/possible_js.shape[0],1).item()


    upper_bound=min(jj+T_u//3,traj_batch.shape[1])
    subsequence=traj_batch[:,jj:upper_bound,:]



    cluster_centers_ids=batch[2][:,jj:upper_bound,:]
    cluster_center_coords=torch.zeros(BB,NN,act_dims)
    for a_i in range(act_dims):

        centers_i=kmeans_lst[a_i].cluster_centers_[cluster_centers_ids[:,:,a_i]]
        cluster_center_coords[:,:,[a_i]]=torch.Tensor(centers_i)
    
    bd_tensor=subsequence[:,:,:bd_dims]
    obs_tensor=subsequence[:,:,bd_dims:bd_dims+obs_dims]
    act_tensor=subsequence[:,:,bd_dims+obs_dims:bd_dims+obs_dims+act_dims] + cluster_center_coords
    subseq_timestamps=torch.arange(jj,upper_bound)

    dbg=False
    if dbg:
        print(colored(f"[DBG] context_size={context_size}, T_text={T_text}, T_u={T_u}","red",attrs=["bold"]))
        print(colored(f"[DBG] jj={jj}, T_u//3={T_u//3}","red",attrs=["bold"]))
        #print(batch[0])

    if input_normalizer is not None:
        bd_tensor, obs_tensor=input_normalizer(
                bd_tensor=bd_tensor,
                obs_tensor=obs_tensor)

    return (text_token_ids.to(device),
            text_posional_ids.to(device),
            bd_tensor.to(device).round(decimals=1),#no need for more precision
            obs_tensor.to(device),
            act_tensor.to(device),#given as model input
            cluster_centers_ids.long().to(device),#useful for supervision
            cluster_center_coords.to(device),#useful for supervision
            subseq_timestamps.to(device),
            )


class GPT_QDRL(nn.Module):

    def __init__(self, config:GPT_QDRLConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            word_token_embedding = nn.Embedding(config.vocab_size, config.n_embd), 
            word_pos_embedding = nn.Embedding(config.block_size, config.n_embd), 
            timestamp_embedding = nn.Embedding(1000, config.n_embd), #those are episode timesteps the 1000 here should be env.max_steps, TODO: don't hardcode that
            bd_embedding=RLMLP(
                config.n_bd_dims,
                h_sz=config.n_embd,
                out_sz=config.n_embd,
                dropout=False),
            obs_embedding=RLMLP(
                config.n_obs_dims, 
                h_sz=config.n_embd,
                out_sz=config.n_embd,
                dropout=False),
            act_embedding=RLMLP(
                config.n_action_dims,
                h_sz=config.n_embd,
                out_sz=config.n_embd,
                dropout=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.action_cluster_heads=nn.ModuleList()
        for a_i in range(self.config.n_action_dims):
            self.action_cluster_heads.append(
                    RLMLP(
                        in_sz=config.n_embd, 
                        h_sz=32,
                        out_sz=config.kmeans_obj_lst[a_i].cluster_centers_.shape[0],
                        scale_out_put=-1,
                        dropout=False)#adding dropout to the last layer would be extremly dumb
                    )

        self.num_clusters=np.prod([x.cluster_centers_.shape[0] for x in self.config.kmeans_obj_lst])
        self.action_prediction_head_offsets=RLMLP(
                in_sz=config.n_embd,
                h_sz=512,
                out_sz=self.num_clusters*config.n_action_dims,
                scale_out_put=-1,
                dropout=False)#adding droupout to the last layer would be extremely dumb

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.term_ratio_hist=[]

    def reconfigure_heads(self, cfg):

        if cfg["cluster_idx_heads"]:
            self.action_cluster_heads=nn.ModuleList()
            for a_i in range(self.config.n_action_dims):
                self.action_cluster_heads.append(
                        RLMLP(
                            in_sz=self.config.n_embd, 
                            h_sz=cfg["cluster_idx_heads"],
                            out_sz=self.config.kmeans_obj_lst[a_i].cluster_centers_.shape[0],
                            scale_out_put=-1,
                            dropout=False)
                        )
            
            print(colored("[WARNING] cluster_idx_heads were re-initialized with random weights, as specified in yaml config.","red",attrs=["bold"]))

        if cfg["cluster_offset_head"]:
            self.action_prediction_head_offsets=RLMLP(
                in_sz=self.config.n_embd,
                h_sz=cfg["cluster_offset_head"],
                out_sz=self.num_clusters*self.config.n_action_dims,
                scale_out_put=-1,
                dropout=False)
            
            print(colored("[WARNING] cluster_offset_head was re-initialized with random weights, as specified in yaml config.","red",attrs=["bold"]))

        return self

    def get_num_params(self):
        """
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
            word_idx,
            word_pos,
            bd_tensor,
            obs_tensor,
            act_tensor,
            cluster_centers_ids, #useful for supervision
            cluster_center_coords, #useful for supervision
            timestamp_tensor,
            generation_mode:bool,
            epoch:int=-1,
            loss_type:str="multimodal_0",
            generation_strategy="sample"):
        """
        Args
            word_idx (torch.tensor): token indexes as returned by the tokenizer, shape (B, T_{text})
            word_pos (torch.tensor): word position indexes, shape T_{text}
            bd_tensor (torch.tensor): behavior descriptors bd_j, ..., bd_{j+LL}. Of shape (B, LL, bd_dims)
            obs_tensor (torch.tensor): observations obs_j, ..., obs_{j+LL}. Of shape (B, LL, obs_dims)
            act_tensor (torch.tensor): actions act_j, ..., act_{j+LL}. Of shape (B, LL, act_dims)
            timestamp_tensor (torch.tensor): timestamps j, ..., {j+LL} corresponding to the subtrajectory
            generation_mode (bool): if True, indicates that we don't care about attention prediction except at the last layer. Note that this in general different from
                                    val/test mode, as we still often want to compute the loss at each step for those splits. It could be seen as a special case of test mode.
            epoch (int): debug variable
            loss_type (str): only "multimodal_0" is available. MSS was completely stupid for this problem and was removed
            generation_strategy (str): only used in generation mode, indicates whether the next action is sampled from the multimodal distribution or if it is its argmax. Valid values are
                                       "argmax" and "sample".
        """

        BB=word_idx.shape[0]
        LL=bd_tensor.shape[1]#subtrajectory length

        word_token_emb= self.transformer.word_token_embedding(word_idx) # (B, T_text, embd_sz)
        word_pos_emb = self.transformer.word_pos_embedding(word_pos) # (T_text, embd_sz)

        bd_emb=self.transformer.bd_embedding(bd_tensor)
        obs_emb=self.transformer.obs_embedding(obs_tensor)
        act_emb=self.transformer.act_embedding(act_tensor)
        timestamp_rep=timestamp_tensor.reshape(-1,1).repeat(1,3).reshape(-1)#each timestamp i is used for three consecutive tokens bd_i, obs_i, act_i (note that padding hasn't happened yet)
        timestamp_emb=self.transformer.timestamp_embedding(timestamp_rep)

        emb_sz=self.config.n_embd
        RL_emb=torch.cat([bd_emb, obs_emb, act_emb],-1).view(BB,3*LL,emb_sz)#3*LL==3*(T_u//3), see the docstring of process_batch for T_u's definition

        
        RL_emb_dbg=torch.cat([bd_emb, obs_emb, act_emb],-1)#(BB, LL, n_embd*3)
        for ii in range(LL):
            assert (RL_emb_dbg[:,ii,0:emb_sz]==RL_emb[:,ii*3,:]).all()
            assert (RL_emb_dbg[:,ii,emb_sz:2*emb_sz]==RL_emb[:,ii*3+1,:]).all()
            assert (RL_emb_dbg[:,ii,2*emb_sz:3*emb_sz]==RL_emb[:,ii*3+2,:]).all()
        
        x1=word_token_emb+word_pos_emb
        x2=RL_emb+timestamp_emb


        xx=torch.cat([x1, x2],1)#(B,context_length-T_u%3,emb_sz), see the docstring of process_batch for T_u's defintion

        #padding if necessary
        num_pad=self.config.block_size-xx.shape[1]
        padding_tensor=torch.zeros(BB,self.config.block_size-xx.shape[1],emb_sz).to(xx.device)

        xx=torch.cat([xx, padding_tensor],1)

        xx = self.transformer.drop(xx)
        for block in self.transformer.h:
            xx = block(xx)
        
        xx = self.transformer.ln_f(xx)
        
        #get observation embeddings
        obs_inds=torch.arange(word_token_emb.shape[1]+1,self.config.block_size-num_pad,step=3).to(xx.device)

        if not generation_mode:


            zz=xx[:,obs_inds,:]

           
            if loss_type=="multimodal_0":

                #### focal term
                term_1=0
                argmax_actions=[]#for accuracy computation
                for a_i in range(self.config.n_action_dims):
                
                    #### focal term i
                    predicted_actions_scores_i=self.action_cluster_heads[a_i](zz) #(BB,LL,num_clusters)
                    gamma=2.0
                    proba_mat_i=torch.softmax(predicted_actions_scores_i,dim=-1)
                    p_t_i=proba_mat_i.gather(2,cluster_centers_ids[:,:,[a_i]])
                    focal_vals_i=-((1-p_t_i)**gamma)*torch.log(p_t_i+1e-8)
                    term_1+=focal_vals_i.mean()

                    #bookkeeping for accuracy metric
                    argmax_act_i=predicted_actions_scores_i.argmax(dim=-1).reshape(BB,LL,1)
                    argmax_actions.append(argmax_act_i)
                
                term_1/=float(self.config.n_action_dims)
          

                #### MSE term
                predicted_actions_offsets=self.action_prediction_head_offsets(zz).reshape(BB,LL,self.num_clusters,-1)#(BB, LL, num_clusters, act_dims)

                #fetch predicted offset for ground truth cluster ids and minimize their MSE
                UU=predicted_actions_offsets # (BB, LL, num_clusters, act_dims)
                VV=cluster_centers_ids.unsqueeze(-2) # (BB, LL, 1, act_dims)
                WW=UU.gather(2, VV)
               
                hat_act=cluster_center_coords+WW.squeeze(-2)
                
                diff=hat_act-act_tensor.round(decimals=1)

                
                #err_thresh=0.05
                #mask=torch.abs(diff)>err_thresh
                #filtered_diff=mask*diff
                #term_2=(filtered_diff**2).mean()#same as MSELoss with "mean" reduction
                
                term_2=(diff**2).mean()#same as MSELoss with "mean" reduction
               
                #print("term_1==",term_1, "term_2==",term_2)
                #self.term_ratio_hist.append(term_1.item()/term_2.item())
                #print("ratio==",np.mean(self.term_ratio_hist))
                
                loss_coef=1.0 #ratio is about 1.43 on average
                loss=term_1+loss_coef*term_2

                term_1_value=term_1.item()
                term_2_value=term_2.item()

                ### accuracy
                #we do the same as for the mse, but using predicted ground truth centers instead
                argmax_actions=torch.cat(argmax_actions,2) # (BB, LL, act_dims)
                WW_acc=UU.gather(2, argmax_actions.unsqueeze(-2))
             
                centers_pred=[]
                for a_i in range(self.config.n_action_dims):
                    center_i=torch.Tensor(self.config.kmeans_obj_lst[a_i].cluster_centers_[argmax_actions[:,:,a_i].detach().cpu().numpy()]).to(UU.device)
                    centers_pred.append(center_i)

                centers_pred=torch.cat(centers_pred,-1)
                action_pred=centers_pred+WW_acc.squeeze(-2)

                diff_ac=action_pred.squeeze(2)-act_tensor.round(decimals=1)
                
                #mask_acc=torch.abs(diff_ac)>err_thresh
                #filtered_diff_acc=mask_acc*diff_ac
                #accuracy=(filtered_diff_acc**2).mean().item()
                accuracy=(diff_ac**2).mean().item()

                #see_tensor=torch.cat([act_tensor,action_pred],-1).detach().cpu().numpy()
                #print(see_tensor)

                predicted_actions=None

            else:
                raise Exception("Unknown loss type")

        else:
            loss=None
            term_1_value=None
            term_2_value=None
            accuracy=None
            zz=xx[:,[obs_inds[-1]],:]

            sampled_acts=[]
            temperature=1.0

            #print(colored(f"generation_strategy={generation_strategy}","magenta",attrs=["bold"]))
            for a_i in range(self.config.n_action_dims):
                #this softmax is kept as we want probabilities to sample from, not a loss
                predicted_actions_scores_i=torch.softmax(self.action_cluster_heads[a_i](zz)/temperature,dim=-1)#(BB, LL, num_clusters_i)

                #sample clusters
                if generation_strategy=="sample":
                    sampled_act_i=torch.multinomial(predicted_actions_scores_i.reshape(-1,self.config.kmeans_obj_lst[a_i].cluster_centers_.shape[0]),num_samples=BB)
                    sampled_act_i=sampled_act_i.reshape(BB,1,1)

                    #print("sampled_act_i==",sampled_act_i)
                    #plt.bar(range(predicted_actions_scores_i.shape[-1]), predicted_actions_scores_i.flatten().detach().cpu().numpy())
                    #plt.show()
                elif generation_strategy=="argmax":
                    sampled_act_i=predicted_actions_scores_i.argmax(dim=-1).reshape(BB,1,1)
                elif generation_strategy=="nucleus":
                    decreasing_scores_idx=predicted_actions_scores_i.flatten().argsort(descending=True).tolist()
                    target_mass=0.95
                    cur_mass=0
                    chosen_inds=[]
                    for s_i in decreasing_scores_idx:
                        chosen_inds.append(s_i)
                        cur_mass+=predicted_actions_scores_i[0, 0, s_i]
                        if cur_mass>=target_mass:
                            break
                    #pdb.set_trace()
                    sampled_act_i=chosen_inds[torch.multinomial(predicted_actions_scores_i[:,:,chosen_inds].flatten()/predicted_actions_scores_i.sum(),num_samples=BB).item()]
                    sampled_act_i=torch.Tensor([sampled_act_i]).reshape(BB,1,1).to(predicted_actions_scores_i.device).long()
                else:
                    raise Exception("Unknown generation strategy")
                sampled_acts.append(sampled_act_i)
            sampled_acts=torch.cat(sampled_acts,-1)
            #fetch corresponding offset predictions
            predicted_actions_offsets=self.action_prediction_head_offsets(zz).reshape(BB,1,self.num_clusters,-1)#(BB, LL, num_clusters, act_dims)
            UU=predicted_actions_offsets
            VV=sampled_acts.unsqueeze(-2) # (BB, 1, act_dims, 1)
            WW=UU.gather(2, VV)
            
           
            #fetch corresponding center coordinates and predict action
            centers_pred=[]
            for a_i in range(self.config.n_action_dims):
                c_i=torch.Tensor(self.config.kmeans_obj_lst[a_i].cluster_centers_[sampled_acts[:,:,a_i].detach().cpu().numpy()]).to(UU.device)
                centers_pred.append(c_i)
           
            centers_pred=torch.cat(centers_pred,-1)
            predicted_actions=centers_pred+WW.squeeze(2)

        return predicted_actions, loss, term_1_value, term_2_value, accuracy

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class QDRLPolicy:

    def __init__(self,
            model:GPT_QDRL,
            tokenizer,
            device,
            input_normalizer=None,
            generation_strategy="sample",
            max_len_pad=226):
        
        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.input_normalizer=input_normalizer
        self.traj_window=None
        self.traj_start_timestamp=0
        self.max_len_pad=max_len_pad
        self.generation_strategy=generation_strategy
    
    def reset(self, prompt_text, prompt_bd):
       
        self.textual_conditioning=prompt_text
        self.target_bd=prompt_bd
        self.traj_window=None
        self.traj_start_timestamp=0

    @property
    def cfg(self):
        return self.model.config

    @torch.no_grad()
    def __call__(self,obs):
        """
        
        Args 
            obs (torch.tensor): shape (1, obs_dims)
        Returns
            action to perform in environment
        Notes

             This function should be optimized as it recomputes all embeddings each time it is called
        """

        self.model.eval()
       
        dummy_action=torch.zeros(1,self.cfg.n_action_dims)#padding for compatibility, it has no effect on the prediction
        new_input=torch.cat([self.target_bd, torch.tensor(obs).reshape(1,-1), dummy_action],1).float()

        self.traj_window=torch.cat([self.traj_window, new_input],0) if self.traj_window is not None else new_input # (LL, DD)


        text_token_ids=self.tokenizer([self.textual_conditioning], padding="max_length", max_length=self.max_len_pad, return_tensors="pt").input_ids 
        T_text=num_text_tokens=text_token_ids.shape[1]
        assert self.cfg.block_size-T_text>10, "The textual prompt is too long. Double check your data, or increase the context length (see yaml config file)"
        text_posional_ids=torch.arange(T_text,dtype=torch.long)

        LL=self.traj_window.shape[0]
        if T_text+3*LL>self.model.config.block_size:
            raise Exception("This should not happen.")

        bd_tensor=self.traj_window[:,:self.cfg.n_bd_dims].unsqueeze(0)
        obs_tensor=self.traj_window[:,self.cfg.n_bd_dims:self.cfg.n_bd_dims+self.cfg.n_obs_dims].unsqueeze(0)
        act_tensor=self.traj_window[:,self.cfg.n_bd_dims+self.cfg.n_obs_dims:self.cfg.n_bd_dims+self.cfg.n_obs_dims+self.cfg.n_action_dims].unsqueeze(0)
        traj_timestamps=torch.arange(self.traj_start_timestamp,self.traj_start_timestamp+LL)
        

        if self.input_normalizer is not None:
            bd_tensor, obs_tensor=self.input_normalizer(
                    bd_tensor=bd_tensor,
                    obs_tensor=obs_tensor)
            bd_tensor=bd_tensor.round(decimals=1)#we do this in train, don't remove it here


        predicted_actions, _ , _, _, _=self.model(
                word_idx=text_token_ids.to(self.device),
                word_pos=text_posional_ids.to(self.device),
                bd_tensor=bd_tensor.to(self.device),
                obs_tensor=obs_tensor.to(self.device),
                act_tensor=act_tensor.to(self.device),
                cluster_centers_ids=None,#this will be predicted
                cluster_center_coords=None,#also predicted
                timestamp_tensor=traj_timestamps.to(self.device),
                generation_mode=True,
                generation_strategy=self.generation_strategy)

        self.traj_window[-1,-2:]=predicted_actions

        return predicted_actions.flatten().tolist()



