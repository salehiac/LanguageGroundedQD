import pickle
import yaml
import argparse
import numpy as np
import random
import functools
import pdb
import json
import matplotlib.pyplot as plt

import torch
from transformers import PreTrainedTokenizerFast
from termcolor import colored
import tqdm

import nanoGPT_QDRL
from dataset_tools import ArchDataset
import MiscUtils

def main_train(
        model,
        train_loader,
        val_loader,
        cfg,
        context_length,
        input_dims,
        tokenizer,
        input_normalizer,
        device,
        log_dir):
    """
    train and validation
    """

    train_val_log_path=MiscUtils.create_directory_with_pid(log_dir+"/train_val_log_",remove_if_exists=True,no_pid=False)
    print(colored(f"Created train_val log directory: {train_val_log_path}","magenta",attrs=["bold"]))

    learning_rate=float(cfg["adamW"]["learning_rate"])
    optimizer=model.configure_optimizers(
            float(cfg["adamW"]["weight_decay"]),
            learning_rate,
            (float(cfg["adamW"]["beta1"]),float(cfg["adamW"]["beta2"])),
            device_type=device.type)

    num_train_steps=0#steps, not epochs
    train_loss_hist=[]
    val_loss_hist=[]
    best_val_loss_idx=-1
    best_val_loss=float("inf")
    for epoch_i in tqdm.tqdm(range(cfg["max_epochs"]),desc="epochs"):

        #print("NUM_TRAIN_STEPS==",num_train_steps)
        
        ##training loop
        model.train()
        train_loss_epc=[]
        lr=MiscUtils.get_lr(it=num_train_steps,
                warmup_iters=cfg["schedule"]["warmup_steps"],
                lr_decay_iters=cfg["schedule"]["lr_decay_steps"],
                learning_rate=learning_rate,
                min_lr=float(cfg["schedule"]["min_lr"])) if cfg["schedule"]["decay_lr"] else learning_rate
 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch in tqdm.tqdm(train_loader,desc="epoch progress (train)",leave=False):

            optimizer.zero_grad()
            processed_batch=nanoGPT_QDRL.process_batch(
                    batch=batch,
                    tokenizer=tokenizer, 
                    context_size=context_length,
                    bd_dims=input_dims["bd"],
                    obs_dims=input_dims["obs"],
                    act_dims=input_dims["act"],
                    device=_device,
                    input_normalizer=_input_normalizer)

            #(text_token_ids, 
            #        text_posional_ids,
            #        bd_tensor,
            #        obs_tensor,
            #        act_tensor,
            #        subseq_timestamps
            #        )=processed_batch

            predicted_actions, loss=model(*processed_batch,generation_mode=False)
            loss.backward()
            optimizer.step()
            num_train_steps+=1

            train_loss_epc.append(loss.item())

            if cfg["adamW"]["grad_clip"]!=0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["adamW"]["grad_clip"])
        train_loss_hist.append(np.mean(train_loss_epc))

 
        ##val loop (no hyperparam optimization here, the val loss is just used to chose a model at the end)
        if epoch_i%cfg["val_frequ"]==0:
            with torch.no_grad():
                model.eval()
                val_loss_epc=[]
                for batch_val in tqdm.tqdm(val_loader,desc="epoch progress (val)",leave=False):
                    processed_batch_val=nanoGPT_QDRL.process_batch(
                        batch=batch_val,
                        tokenizer=tokenizer, 
                        context_size=context_length,
                        bd_dims=input_dims["bd"],
                        obs_dims=input_dims["obs"],
                        act_dims=input_dims["act"],
                        device=_device,
                        input_normalizer=_input_normalizer)
                
                    predicted_actions_val, loss_val=model(*processed_batch_val,generation_mode=False)
                    val_loss_epc.append(loss_val.item())
                val_loss_epc_mean=np.mean(val_loss_epc)
                if val_loss_epc_mean<best_val_loss:
                    best_val_loss=val_loss_epc_mean
                    best_val_loss_idx=epoch_i
        
        #we still add the val_loss_epc_mean even epoch_i%val_frequ!=0. This is just for display, so that we get the same number of inputs to plt.plot as for train, without interpolation
        val_loss_hist.append(val_loss_epc_mean)
        
        torch.save(model,train_val_log_path+f"/model_{epoch_i}")

        with open(train_val_log_path+f"/progress_info_{epoch_i}","w") as fl:
            dd={"train_loss":train_loss_hist,"val_loss": val_loss_hist,"epoch_with_best_val_loss": best_val_loss_idx,"lr":lr}
            json.dump(dd,fl)

    plt.plot(train_loss_hist,"r")
    plt.plot(val_loss_hist,"b")
    plt.show()

def main_test(
        model,
        test_loader,
        cfg,
        context_length,
        input_dims,
        tokenizer,
        input_normalizer,
        device,
        log_dir):
    """
    test
    """

    test_log_path=MiscUtils.create_directory_with_pid(log_dir+"/test_log_",remove_if_exists=True,no_pid=False)
    print(colored(f"Created test_log directory: {test_log_path}","magenta",attrs=["bold"]))

    model.eval()
    with torch.no_grad():
        test_loss=[]
        for batch_test in tqdm.tqdm(test_loader,desc="test",leave=False):
            processed_batch_test=nanoGPT_QDRL.process_batch(
                    batch=batch_test,
                    tokenizer=tokenizer, 
                    context_size=context_length,
                    bd_dims=input_dims["bd"],
                    obs_dims=input_dims["obs"],
                    act_dims=input_dims["act"],
                    device=_device,
                    input_normalizer=_input_normalizer)

            predicted_actions_test, loss_test=model(*processed_batch_test,generation_mode=False)
            test_loss.append(loss_test.item())

        plt.plot(test_loss)
        plt.title(np.mean(test_loss))
        plt.show()


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

    if _config["train_model"] or _config["test_model"]:
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

    _input_normalizer=None
    _nav_env=None
    if any(_config["input_normalization"]["normalize"]):
        if _config["input_normalization"]["env_type"]=="navigation_env":
            from dataset_tools import make_navigation_env
            _nav_env=make_navigation_env()
            _input_normalizer=functools.partial(_nav_env.normalize_bd_obs_act,options=_config["input_normalization"]["normalize"],dbg=True)
        else: 
            raise NotImplementedError("Only available env is navigation_env")

    if _config["train_model"]:
        _out_train=main_train(
                model=_model,
                train_loader=_train_loader,
                val_loader=_val_loader,
                cfg=_config["train_cfg"],
                context_length=_config["model_cfg"]["block_size"],
                input_dims={"bd":_bd_dims, "obs":_obs_dims, "act":_cmd_dims},
                tokenizer=_tokenizer,
                input_normalizer=_input_normalizer,
                device=_device,
                log_dir=_config["logging"]["log_dir"])
    if _config["test_model"]:
        main_test(
                model=_model,
                test_loader=_test_loader,
                cfg=_config["test_cfg"],
                context_length=_config["model_cfg"]["block_size"],
                input_dims={"bd":_bd_dims, "obs":_obs_dims, "act":_cmd_dims},
                tokenizer=_tokenizer,
                input_normalizer=_input_normalizer,
                device=_device,
                log_dir=_config["logging"]["log_dir"])

    if _config["deploy_in_env"]:
        if _config["deploy_cfg"]["env_type"]!="navigation_env":
            raise NotImplementedError("Only available env is navigation_env")
        if _nav_env is None:
            from dataset_tools import make_navigation_env
            _nav_env=make_navigation_env()

        _load_prompts=lambda x: (json.load(fl:=open(x,"r")),fl.close())[0]
        prompt_lst=_load_prompts(_config["deploy_cfg"]["prompts"])

        policy=nanoGPT_QDRL.QDRLPolicy(
                _model,
                tokenizer=_tokenizer,
                device=_device,
                input_normalizer=_input_normalizer,
                use_default_start=-1)

        for prompt_i in range(len(prompt_lst)):
            prompt=prompt_lst[prompt_i]
            print(colored(f"generating trajectory for prompt {prompt_i}...","green",attrs=["bold"]))
            policy.reset(prompt_text=prompt[0],
                    prompt_bd=torch.tensor(prompt[1:1+_nav_env.get_bd_dims()]).reshape(1,2))

            (fitness,
                    tau_np,
                    behavior2d_np,
                    bd,
                    task_solved
                    )=_nav_env(policy)

            dummy_cls = type('dummy_cls', (object,), {'_behavior': behavior2d_np})
            _nav_env.visualise_behavior(dummy_cls,hold_on=True)
            plt.title(prompt[0])
            plt.show()



 
    debug=False
    if debug:
        _train_loader_it=iter(_train_loader)
        _bb=next(_train_loader_it)

       
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

