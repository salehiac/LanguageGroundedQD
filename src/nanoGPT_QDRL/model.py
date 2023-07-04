"""
This code extends nanoGPT (https://github.com/karpathy/nanoGPT) for conditioning on language and behavior descriptors
"""

import math
import inspect
from collections import namedtuple
from typing import Literal, List
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from termcolor import colored

import nanoGPT_QDRL.QDRLTokenWindow as QDRLTokenWindow

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

class RLMLP(torch.nn.Module):
    """
    the main reason this is separate from the MLP class here is that the MLP module is used in residual layers  
    and its c_fc is initialized differently because of that. I didn't want to change that behavior from the original repo
    """

    def __init__(self, in_sz, emb_sz, dropout=0.0, bias=True, scale_out_put=-1):
        
        super().__init__()
        self.l1=torch.nn.Linear(in_sz, 2*emb_sz, bias)
        self.l2=torch.nn.Linear(2*emb_sz, emb_sz, bias)
        self.l3=torch.nn.Linear(emb_sz, emb_sz, bias)
        self.nonlin=torch.nn.GELU()
        self.dropout=torch.nn.Dropout(dropout)
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
    ])

   
def process_batch(
        batch,
        tokenizer,
        context_size,
        bd_dims,
        obs_dims,
        act_dims,
        device,
        input_normalizer=None,
        max_len_pad=226):
    """
    Args: 
        batch (list): a list of length 2, with
                                        - batch[0] a list of batch_size strings
                                        - batch[1] a tensor of shape batch_size*M, with M=ep_len*(bd_dims+obs_dims+act_dims)
                                         each example batch[ex,:] is a 1D tensor that semantically can be separated into
                                        [ bd_0, obs_0, act_0,
                                          bd_1, obs_1, act_1,
                                          ...
                                          bd_{N-1], obs_{N-1}, act_{N-1}] #with N the episode length. Note that all example trajectories are assumed to have the same length
                                                                          #Note that each of the bd_i, obs_i, act_i are considered as a separate token, so 
                                                                          #there are a total of 3N tokens in each trajectory

                                        with bd_i, obs_i, act_i respectively of lengths bd_dims, obs_dims and act_dims.
        
        tokenizer (PreTrainedTokenizerFast): A tokenizer pretrained on the corpus
        context_size (int): the context size of the transformer
        bd_dims (int): length of behavior descriptors
        obs_dims (int): length of observations vector
        act_dims (int): length of action vector

    Returns:

        text_token_ids (torch.LongTensor): shape batch_size*T_text, with T_text the number of tokkens after padding to the token length of the longest string in the batch.
        text_posional_ids (torch.LongTensor): 1d tensor of shape T_text, to compute positional embeddings.
        bd_tensor (torch.tensor): shape batch_size*T_u*bd_dims. See the notes section below for the defintion of T_u.
        obs_tensor (torch.tensor): shape batch_size*T_u*obs_dims. See the notes section below for the defintion of T_u.
        act_tensor (torch.tensor): shape batch_size*T_u*act_dims. See the notes section below for the defintion of T_u.
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
    #text_token_ids=tokenizer(text_batch, padding=True, return_tensors="pt").input_ids #padding might not be the most optimal way, but it simplifies things
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


    bd_tensor=subsequence[:,:,:bd_dims]
    obs_tensor=subsequence[:,:,bd_dims:bd_dims+obs_dims]
    act_tensor=subsequence[:,:,bd_dims+obs_dims:bd_dims+obs_dims+act_dims]
    subseq_timestamps=torch.arange(jj,upper_bound)



    #QDRLTokenWindow is just useful for debug and will be removed in subsequent updates
    #tw=QDRLTokenWindow.QDRLTokenWindow(subsequence.reshape(BB,-1),bd_dims,obs_dims,act_dims,jj)
    #bd_tensor_dbg=tw.get_bd_tensor()
    #obs_tensor_dbg=tw.get_obs_tensor()
    #act_tensor_dbg=tw.get_act_tensor()
    #assert (bd_tensor_dbg==bd_tensor).all()
    #assert (obs_tensor_dbg==obs_tensor).all()
    #assert (act_tensor==act_tensor_dbg).all()
    #assert (subseq_timestamps==tw.get_position_tensor()).all()

    #pdb.set_trace()
    
    dbg=False
    if dbg:
        print(colored(f"[DBG] context_size={context_size}, T_text={T_text}, T_u={T_u}","red",attrs=["bold"]))
        print(colored(f"[DBG] jj={jj}, T_u//3={T_u//3}","red",attrs=["bold"]))
        #print(batch[0])

    if input_normalizer is not None:
        bd_tensor, obs_tensor, act_tensor=input_normalizer(
                bd_tensor=bd_tensor,
                obs_tensor=obs_tensor,
                act_tensor=act_tensor)
    
    
    return (text_token_ids.to(device),
            text_posional_ids.to(device),
            bd_tensor.to(device).round(decimals=1),#no need for more precision
            obs_tensor.to(device),
            act_tensor.to(device),
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
            bd_embedding=RLMLP(config.n_bd_dims, config.n_embd),
            obs_embedding=RLMLP(config.n_obs_dims, config.n_embd),
            act_embedding=RLMLP(config.n_action_dims, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        #self.action_prediction_head=nn.Linear(config.n_embd,config.n_action_dims)
        self.action_prediction_head=RLMLP(config.n_embd,config.n_action_dims,scale_out_put=-1)#, dropout=config.dropout)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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
            timestamp_tensor,
            generation_mode:bool,
            epoch:int=-1):
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
            predicted_actions=self.action_prediction_head(zz)#(B, LL, act_dims)

            #compute loss
            loss=((predicted_actions-act_tensor.round(decimals=1))**2).mean()#same as MSELoss with "mean" reduction

            #pdb.set_trace()

            to_see=torch.cat([act_tensor.round(decimals=1),predicted_actions],-1)
            print(to_see)
            #if epoch%1000==0 and epoch:
            #    pdb.set_trace()
        else:
            zz=xx[:,[obs_inds[-1]],:]
            predicted_actions=self.action_prediction_head(zz) #(B, 1, act_dims)
            loss=None

        return predicted_actions, loss

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
            use_default_start=-1):
        
        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.input_normalizer=input_normalizer
        self.use_default_start=use_default_start
        self.traj_window=None
        self.traj_start_timestamp=0
    
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


        text_token_ids=self.tokenizer([self.textual_conditioning], padding=True, return_tensors="pt").input_ids
        T_text=num_text_tokens=text_token_ids.shape[1]
        assert self.cfg.block_size-T_text>10, "The textual prompt is too long. Double check your data, or increase the context length (see yaml config file)"
        text_posional_ids=torch.arange(T_text,dtype=torch.long)

        LL=self.traj_window.shape[0]
        if T_text+3*LL>self.model.config.block_size:
            num_drop=2
            self.traj_window=self.traj_window[num_drop:,:]
            LL-=num_drop
            self.traj_start_timestamp+=num_drop

        bd_tensor=self.traj_window[:,:self.cfg.n_bd_dims].unsqueeze(0)
        obs_tensor=self.traj_window[:,self.cfg.n_bd_dims:self.cfg.n_bd_dims+self.cfg.n_obs_dims].unsqueeze(0)
        act_tensor=self.traj_window[:,self.cfg.n_bd_dims+self.cfg.n_obs_dims:self.cfg.n_bd_dims+self.cfg.n_obs_dims+self.cfg.n_action_dims].unsqueeze(0)
        traj_timestamps=torch.arange(self.traj_start_timestamp,self.traj_start_timestamp+LL)
        

        if self.input_normalizer is not None:
            bd_tensor, obs_tensor, act_tensor=self.input_normalizer(
                    bd_tensor=bd_tensor,
                    obs_tensor=obs_tensor,
                    act_tensor=act_tensor)
        
        #pdb.set_trace()
        
        predicted_actions, _ =self.model(
                text_token_ids.to(self.device),
                text_posional_ids.to(self.device),
                bd_tensor.to(self.device),
                obs_tensor.to(self.device),
                act_tensor.to(self.device),
                traj_timestamps.to(self.device),
                generation_mode=True)

        self.traj_window[-1,-2:]=predicted_actions

        return predicted_actions.flatten().tolist()



