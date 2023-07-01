import subprocess
import os
import sys
from datetime import datetime
import functools
import pdb
import random
from dataclasses import dataclass
import numpy as np
from functools import reduce
import string
import re
import math
import torch
import matplotlib.pyplot as plt

sys.path.append("../")

def get_current_time_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def rand_string(alpha=True, numerical=True):
    l2="0123456789" if numerical else ""
    return reduce(lambda x,y: x+y, random.choices(string.ascii_letters+l2,k=10),"")

def bash_command(cmd:list):
    """
    cmd  list [command, arg1, arg2, ...]
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ret_code=proc.returncode

    return out, err, ret_code

def create_directory_with_pid(dir_basename,remove_if_exists=True,no_pid=False):
    while dir_basename[-1]=="/":
        dir_basename=dir_basename[:-1]
    
    dir_path=dir_basename+str(os.getpid()) if not no_pid else dir_basename
    if os.path.exists(dir_path):
        if remove_if_exists:
            bash_command(["rm",dir_path,"-rf"])
        else:
            raise Exception("directory exists but remove_if_exists is False")
    bash_command(["mkdir", dir_path])
    notif_name=dir_path+"/creation_notification.txt"
    bash_command(["touch", notif_name])
    with open(notif_name,"w") as fl:
        fl.write("created on "+get_current_time_date()+"\n")
    return dir_path

def selBest(individuals,k,fit_attr=None,automatic_threshold=True):
   
    individual_novs=[x._nov for x in individuals]

    if automatic_threshold:
        md=np.median(individual_novs)
        individual_novs=list(map(lambda x: x if x>md else 0, individual_novs))

    s_indx=np.argsort(individual_novs).tolist()[::-1]#decreasing order
    return [individuals[i] for i in s_indx[:k]]

def normalize_th(x,low, high,scale, return_inverse_function:bool):
    """
    finds ax+b to map x in [low, high] to [-scale,scale]

    if return_inverse_function is True, then the function also returns and f that does the inverse mapping
    """

    m=low
    M=high
    assert m<M
    assert (x<=high).all()
    assert (x>=low).all()

    a=2/(M-m)
    b=-(M+m)/(M-m)
    #print(f'a={a},   b={b}')
    res=a*x+b

    if not return_inverse_function:
        return res*scale, None
    else:
        def inv_f(yy):

            return (yy/scale-b)/a

        return res*scale, inv_f

def get_lr(it,
        warmup_iters,
        lr_decay_iters,
        learning_rate,
        min_lr):
    """
    same as what happends in nanoGPT's original train loop
    learning rate decay scheduler (cosine with warmup)
    """
    if it < warmup_iters:
        return learning_rate * (1+it) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    #in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def plot_planned_scheduling(
        warmup_iters,
        lr_decay_iters,
        learning_rate,
        min_lr):
    mm=[get_lr(_it,warmup_iters,lr_decay_iters,learning_rate,min_lr) for _it in range(100000)]
    plt.plot(mm)
    plt.title("warmup + cosine annealing (no restarts)")
    plt.show()
    plt.close()
 

    
@dataclass
class colors:
    red=(255,0,0)
    green=(0,255,0)
    blue=(0,0,255)
    yellow=(255,255,51)

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def add_newlines(string, chars_per_line=60):
    words = string.split(' ')
    lines = []
    line = ''
    for word in words:
        if len(line + ' ' + word) <= chars_per_line:
            line += ' ' + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return '\n'.join(lines)


if __name__=="__main__":

    test_normalize_th=False
    if test_normalize_th:
        _test_tensor=torch.rand(4,5)*100
        _normalized_tensor,_inv_f=normalize_th(_test_tensor, low=-100,high=100,scale=1.0,return_inverse_function=True)
        _recons=_inv_f(_normalized_tensor)
        print(f"test_tensor\n {_test_tensor}\nnormalized_tensor\n {_normalized_tensor}\n************")
        print(f"inverse function correct? {torch.allclose(_test_tensor, _recons)}\n********************************")

        _test_tensor=torch.rand(4,5)*5+4
        _normalized_tensor, _inv_f=normalize_th(_test_tensor, low=-2,high=16,scale=1.0,return_inverse_function=True)
        print(f"test_tensor\n {_test_tensor}\nnormalized_tensor\n {_normalized_tensor}\n************")
        _recons=_inv_f(_normalized_tensor)
        print(f"inverse function correct? {torch.allclose(_test_tensor, _recons)}\n********************************")
   
    plot_lr_schedule=True
    if plot_lr_schedule:


        _warmup_iters=200#a bit more than one epoch with the toy dataset and batchsize 32
        _lr_decay_iters=76000#with the toy train dataset and batchsize 32, each epoch is made of 162 updates => The first and a half epoch are spent in warmup
                             #say we have 100 epochs => 162000 updates => let's spend 15000 of those decaying
        _learning_rate=5e-3
        _min_lr=5e-5
        plot_planned_scheduling(
                _warmup_iters,
                _lr_decay_iters,
                _learning_rate,
                _min_lr)

   
