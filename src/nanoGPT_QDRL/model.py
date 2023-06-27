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
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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

GPTConfig=namedtuple("GPTConfig",[
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
        device):
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
    text_token_ids=tokenizer(text_batch, padding=True, return_tensors="pt").input_ids #padding might not be the most optimal way, but it simplifies things
    T_text=num_text_tokens=text_token_ids.shape[1]
    text_posional_ids=torch.arange(T_text,dtype=torch.long)
 
    #number of 
    T_u=context_size-T_text

    BB=batch[1].shape[0]
    DD=bd_dims+obs_dims+act_dims
    NN=batch[1].shape[1]//DD #episode length
    traj_batch=batch[1].reshape(BB,NN,DD) #traj_batch[ex_i,j,:] is bd_j, obs_j, act_j

    possible_js=torch.arange(0,NN-T_u//3+1,dtype=torch.long)
    jj=torch.multinomial(torch.ones_like(possible_js).float()/possible_js.shape[0],1).item()

    subsequence=traj_batch[:,jj:,:]


    bd_tensor=subsequence[:,:,:bd_dims]
    obs_tensor=subsequence[:,:,bd_dims:bd_dims+obs_dims]
    act_tensor=subsequence[:,:,bd_dims+obs_dims:bd_dims+obs_dims+act_dims]
    subseq_timestamps=torch.arange(jj,NN)



    #QDRLTokenWindow is just useful for debug and will be removed in subsequent updates
    tw=QDRLTokenWindow.QDRLTokenWindow(subsequence.reshape(BB,-1),bd_dims,obs_dims,act_dims,jj)
    bd_tensor_dbg=tw.get_bd_tensor()
    obs_tensor_dbg=tw.get_obs_tensor()
    act_tensor_dbg=tw.get_act_tensor()
    assert (bd_tensor_dbg==bd_tensor).all()
    assert (obs_tensor_dbg==obs_tensor).all()
    assert (act_tensor==act_tensor_dbg).all()
    assert (subseq_timestamps==tw.get_position_tensor()).all()

    """TODO: don't forget masking for the padded values! The attention's softmax is computed over all values of QK^T, and since your padding will result on garbage values on the few last
    lines, if you don't add -inf to them, they might perturb the softmat"""
    
    return (text_token_ids.to(device),
            text_posional_ids.to(device),
            bd_tensor.to(device),
            obs_tensor.to(device),
            act_tensor.to(device),
            subseq_timestamps.to(device),
            )




class GPT(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #word token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #word position embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)#reshapes into [B*T,vocab_size] and ignores idx==-1, which I think corresponds to [PAD] 
                                                                                                       #or other special tokens
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
