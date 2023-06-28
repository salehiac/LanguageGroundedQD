import pickle
import yaml
import argparse
import numpy as np
import random
import functools

import torch
from transformers import PreTrainedTokenizerFast
from termcolor import colored

import nanoGPT_QDRL
from dataset_tools import ArchDataset

def main_train(cfg,tokenizer,device,log_dir):
    pass

def main_test(cfg,tokenizer,device,log_dir):
    pass


if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='Train/Test the mdoel.')
    _parser.add_argument('--config', type=str,  help="yaml config", default="")

    _args=_parser.parse_args()

    with open(_args.config,"r") as fl:
        _config=yaml.load(fl,Loader=yaml.FullLoader)

    if _config["deterministic"]:
        _seed=127
        print(colored(f"'deterministic' was set to True in the config file, setting seed to {_seed}","magenta",attrs=["bold"]))
        torch.manual_seed(_seed)
        np.random.seed(_seed)
        random.seed(_seed)


    assert _config["train_model"] or _config["pretrained_model"], "You disabled training without specifying a model to test"
    
    torch.set_default_dtype(getattr(torch,_config["dtype"]))
    _device=torch.device("cpu") if not torch.cuda.is_available() else torch.device(_config["device"])

    #load train/val/test archives and create datasets/dataloaders
    _load_archs=lambda fn: (pickle.load(f:=open(fn,"rb")),f.close())[0]
    _arch_train_path=_config["train_cfg"]["data_path_train"]
    _arch_val_path=_config["train_cfg"]["data_path_val"]
    _arch_test_path=_config["test_cfg"]["data_path"]
    _arch_train, _arch_val, _arch_test=[_load_archs(x) for x in [_arch_train_path, _arch_val_path, _arch_test_path]]

    _cmd_dims=_arch_train[0]._tau["action"].shape[1]
    _obs_dims=_arch_train[0]._tau["obs"].shape[1]
    _bd_dims=_arch_train[0]._behavior_descr.shape[1]

    _train_dataset=ArchDataset(_arch_train,split="train")
    _train_loader=_train_dataset.make_data_loader(batch_size=_config["train_cfg"]["batch_size"])

    _val_dataset=ArchDataset(_arch_val,split="val")
    _val_loader=_val_dataset.make_data_loader(batch_size=_config["train_cfg"]["batch_size"])

    _test_dataset=ArchDataset(_arch_test,split="test")
    _test_loader=_test_dataset.make_data_loader(batch_size=_config["test_cfg"]["batch_size"])

    #load tokenizer and create/load model
    _tokenizer=PreTrainedTokenizerFast.from_pretrained(_config["model_cfg"]["learned_tokenizer"])
    if _config["pretrained_model"]:

        _model=torch.load(_config["pretrained_model"])

    else:
        _gpt_cfg=nanoGPT_QDRL.GPT_QDRLConfig(
                block_size=_config["model_cfg"]["block_size"],
                vocab_size=len(_tokenizer.get_vocab()),
                n_layer=_config["model_cfg"]["n_layer"],
                n_head=_config["model_cfg"]["n_head"],
                n_embd=_config["model_cfg"]["n_embd"],
                dropout=_config["model_cfg"]["dropout_p"],
                bias=_config["model_cfg"]["bias"],
                n_action_dims=_cmd_dims,
                n_obs_dims=_obs_dims,
                n_bd_dims=_bd_dims,
                )

        _model=nanoGPT_QDRL.GPT_QDRL(_gpt_cfg)
        _model.to(_device)


    if _config["train_model"]:
        _out_train=main_train(_config["train_cfg"],_tokenizer,_device,log_dir=_config["logging"]["log_dir"])
    if _config["test_model"]:
        _out_train=main_test(_config["test_cfg"],_tokenizer,_device,log_dir=_config["logging"]["log_dir"])

    debug=True
    if debug:
        _train_loader_it=iter(_train_loader)
        _bb=next(_train_loader_it)

        _input_normalizer=None
        if any(_config["input_normalization"]["normalize"]):
            if _config["input_normalization"]["env_type"]=="navigation_env":
                from dataset_tools import make_navigation_env
                _nav_env=make_navigation_env()
                _input_normalizer=functools.partial(_nav_env.normalize_bd_obs_act,options=_config["input_normalization"]["normalize"],dbg=True)
            else: 
                raise NotImplementedError("Only available env is navigation_env")

        _processed_batch=nanoGPT_QDRL.process_batch(batch=_bb,
                tokenizer=_tokenizer, 
                context_size=_config["model_cfg"]["block_size"],
                bd_dims=_bd_dims,
                obs_dims=_obs_dims,
                act_dims=_cmd_dims,
                device=_device,
                input_normalizer=_input_normalizer)
       
        (text_token_ids, 
                text_posional_ids,
                bd_tensor,
                obs_tensor,
                act_tensor,
                subseq_timestamps
                )=_processed_batch

        predicted_actions, loss=_model(*_processed_batch,generation_mode=False)

