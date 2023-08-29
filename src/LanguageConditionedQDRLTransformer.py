import pickle
import yaml
import argparse
import numpy as np
import random
import functools
import json
import matplotlib.pyplot as plt

import torch
from transformers import PreTrainedTokenizerFast
from termcolor import colored
import tqdm
from scoop import futures

import nanoGPT_QDRL
from dataset_tools import ArchDataset
import MiscUtils
from environment import create_env_with_objects

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
        kmeans_lst,
        log_dir):
    """
    train and validation
    """

    train_val_log_path=MiscUtils.create_directory_with_pid(log_dir+"/train_val_log_",remove_if_exists=True,no_pid=False)
    print(colored(f"Created train_val log directory: {train_val_log_path}","magenta",attrs=["bold"]))

    learning_rate=float(cfg["adamW"]["learning_rate"])

    if not any(cfg["only_train"].values()):
        print(colored("[Info] The model will be trained end to end.", "green", attrs=["bold"]))
        optimizer=model.configure_optimizers(
                float(cfg["adamW"]["weight_decay"]),
                learning_rate,
                (float(cfg["adamW"]["beta1"]),float(cfg["adamW"]["beta2"])),
                device_type=device.type)
    else:
        print(colored("[Info] The backbone has been frozen, only some heads will be trained.", "green", attrs=["bold"]))
        param_groups= []

        for param in model.parameters():#to reduce unnecessary memory consumption
            param.requires_grad=False
        
        if cfg["only_train"]["cluster_idx_heads"]:
            for param in model.action_cluster_heads.parameters():
                param.requires_grad=True
            param_groups.append({'params': model.action_cluster_heads.parameters()})#, 'weight_decay': float(cfg["adamW"]["weight_decay"])})
        if cfg["only_train"]["cluster_offset_head"]:
            for param in model.action_prediction_head_offsets.parameters():
                param.requires_grad=True
            param_groups.append({'params': model.action_prediction_head_offsets.parameters()})#, 'weight_decay': float(cfg["adamW"]["weight_decay"])})

        optimizer=torch.optim.Adam(param_groups,lr=learning_rate)

    num_train_steps=0#steps, not epochs
    train_loss_hist=[]
    train_term_1_hist=[]
    train_term_2_hist=[]
    train_acc_hist=[]
    val_loss_hist=[]
    val_term_1_hist=[]
    val_term_2_hist=[]
    val_acc_hist=[]
    best_val_loss_idx=-1
    best_val_acc=float("inf")
    tqdm_epoch=tqdm.tqdm(range(cfg["max_epochs"]),desc="epochs")

    if cfg["schedule"]["decay_lr"]:
        MiscUtils.plot_planned_scheduling(
                cfg["schedule"]["warmup_steps"],
                cfg["schedule"]["lr_decay_steps"],
                learning_rate,
                min_lr=float(cfg["schedule"]["min_lr"]),
                max_plot_range=cfg["schedule"]["lr_decay_steps"]+30000) 


    for epoch_i in tqdm_epoch:

        ##training loop
        model.train()
        train_loss_epc=[]
        train_term_1_epc=[]
        train_term_2_epc=[]
        train_acc_epc=[]
        lr=MiscUtils.get_lr(it=num_train_steps,
                warmup_iters=cfg["schedule"]["warmup_steps"],
                lr_decay_iters=cfg["schedule"]["lr_decay_steps"],
                learning_rate=learning_rate,
                min_lr=float(cfg["schedule"]["min_lr"])) if cfg["schedule"]["decay_lr"] else learning_rate

 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        tqdm_train_loader=tqdm.tqdm(train_loader,desc="epoch progress (train)",leave=False)
        for batch in tqdm_train_loader:

            optimizer.zero_grad()

            processed_batch=nanoGPT_QDRL.process_batch(
                    batch=batch,
                    tokenizer=tokenizer, 
                    context_size=context_length,
                    bd_dims=input_dims["bd"],
                    obs_dims=input_dims["obs"],
                    act_dims=input_dims["act"],
                    kmeans_lst=kmeans_lst,
                    device=_device,
                    input_normalizer=input_normalizer)

            _, loss, term_1, term_2 , accuracy =model(*processed_batch,generation_mode=False,epoch=epoch_i)
            loss.backward()
            optimizer.step()
            num_train_steps+=1

            train_loss_epc.append(loss.item())
            train_term_1_epc.append(term_1)
            train_term_2_epc.append(term_2)
            train_acc_epc.append(accuracy)

            if float(cfg["adamW"]["grad_clip"])!=0.0:
                pdb.set_trace()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["adamW"]["grad_clip"])

            tqdm_train_loader.set_postfix({"batch loss":loss.item()})
        train_loss_hist.append(np.mean(train_loss_epc))
        train_term_1_hist.append(np.mean(train_term_1_epc))
        train_term_2_hist.append(np.mean(train_term_2_epc))
        train_acc_hist.append(np.mean(train_acc_epc))
        tqdm_epoch.set_postfix({"epoch loss":train_loss_hist[-1],"LR":lr})

 
        ##val loop (no hyperparam optimization here, the val loss is just used to chose a model at the end)
        if epoch_i%cfg["val_frequ"]==0:
            with torch.no_grad():
                model.eval()
                val_loss_epc=[]
                val_term_1_epc=[]
                val_term_2_epc=[]
                val_acc_epc=[]
                for batch_val in tqdm.tqdm(val_loader,desc="epoch progress (val)",leave=False):

                    processed_batch_val=nanoGPT_QDRL.process_batch(
                        batch=batch_val,
                        tokenizer=tokenizer, 
                        context_size=context_length,
                        bd_dims=input_dims["bd"],
                        obs_dims=input_dims["obs"],
                        act_dims=input_dims["act"],
                        kmeans_lst=kmeans_lst,
                        device=_device,
                        input_normalizer=input_normalizer)
                
                    _, loss_val, term_1_val, term_2_val, accuracy_val=model(*processed_batch_val,generation_mode=False)
                    val_loss_epc.append(loss_val.item())
                    val_term_1_epc.append(term_1_val)
                    val_term_2_epc.append(term_2_val)
                    val_acc_epc.append(accuracy_val)
                
                val_loss_epc_mean=np.mean(val_loss_epc)
                val_term_1_epc_mean=np.mean(val_term_1_epc)
                val_term_2_epc_mean=np.mean(val_term_2_epc)
                val_acc_epc_mean=np.mean(val_acc_epc)
                if val_acc_epc_mean<best_val_acc:
                    best_val_acc=val_acc_epc_mean
                    best_val_loss_idx=epoch_i
        
        #we still add the val_loss_epc_mean even epoch_i%val_frequ!=0. This is just for display, so that we get the same number of inputs to plt.plot as for train, without interpolation
        val_loss_hist.append(val_loss_epc_mean)
        val_term_1_hist.append(val_term_1_epc_mean)
        val_term_2_hist.append(val_term_2_epc_mean)
        val_acc_hist.append(val_acc_epc_mean)
       
        if best_val_loss_idx==epoch_i:
            torch.save(model,train_val_log_path+f"/model_{epoch_i}")

        torch.save(model,train_val_log_path+f"/last_model")

        with open(train_val_log_path+f"/progress_info_{epoch_i}","w") as fl:
            dd={
                    "train_loss":train_loss_hist,
                    "train_term_1":train_term_1_hist,
                    "train_term_2":train_term_2_hist,
                    "train_acc_hist":train_acc_hist,
                    "val_loss": val_loss_hist,
                    "val_term_1":val_term_1_hist,
                    "val_term_2":val_term_2_hist,
                    "val_acc_hist":val_acc_hist,
                    "epoch_with_best_val_loss": best_val_loss_idx,
                    "lr":lr
                    }
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
        kmeans_lst,
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
        test_loss_term_1=[]
        test_loss_term_2=[]
        for batch_test in tqdm.tqdm(test_loader,desc="test",leave=False):
            processed_batch_test=nanoGPT_QDRL.process_batch(
                    batch=batch_test,
                    tokenizer=tokenizer, 
                    context_size=context_length,
                    bd_dims=input_dims["bd"],
                    obs_dims=input_dims["obs"],
                    act_dims=input_dims["act"],
                    kmeans_lst=kmeans_lst,
                    device=_device,
                    input_normalizer=input_normalizer)

            _, loss_test, term_1_test, term_2_test=model(*processed_batch_test,generation_mode=False)
            test_loss.append(loss_test.item())
            test_loss_term_1.append(term_1_test)
            test_loss_term_2.append(term_2_test)

        plt.plot(test_loss)
        plt.plot(test_loss_term_1)
        plt.plot(test_loss_term_2)
        plt.title(np.mean(test_loss))
        plt.show()

def generate_traj(
        prompt,
        idx,
        basename,
        bd_dims, 
        policy,
        nav_env,
        scene,
        ablate_text:bool,
        ablate_bd:bool,
        num_trials:int):


    out_fn_fig=f"{basename}/fig_{idx}.png"
    out_fn_prompt_annotation_pairs=f"{basename}/pair_{idx}.json"
    out_fn_bds=f"{basename}/bds_{idx}.json"
    bds_lst=prompt[1:1+bd_dims] if not ablate_bd else [0,0]
    prompt_text=prompt[0] if not ablate_text else ""
    best_bd_dist=float("inf")

    bd_dist_lst=[]
    for trial_i in range(num_trials):

        print(colored(f"Trial {trial_i+1}/{num_trials}","magenta",attrs=["bold"]))
        policy.reset(prompt_text=prompt_text,
                prompt_bd=torch.tensor(bds_lst).reshape(1,2))

        (_,_,behavior2d_np_i,bd_i,_)=nav_env(policy)
        dist_i=np.linalg.norm(np.array(bds_lst)/3-bd_i.reshape(2)/3)
        if dist_i<best_bd_dist:
            best_bd_dist=dist_i
            behavior2d_np=behavior2d_np_i
            bd=bd_i
        bd_dist_lst.append(dist_i)#just to compute std
        print("trial bd_dist==",dist_i)

    fig,_=scene.display(display_bbox=False,
            hold_on=True,
            path2d_info=(behavior2d_np, 600, 600))

    annotation = scene.annotate_traj(
            behavior2d_np, real_w=600, real_h=600, step=40)

    fig.suptitle(MiscUtils.add_newlines(prompt_text+f"\n BDs={[x//3 for x in bds_lst]}"))
    plt.tight_layout()
    plt.savefig(out_fn_fig)
    plt.close()

    with open(out_fn_prompt_annotation_pairs, "w") as fl:
        json.dump({"prompt":prompt_text, "predicted_traj_annotation": annotation},fl)

    with open(out_fn_bds,"w") as fl:
        #the factor of 3 is because the env is 200x200 but bds are in 600x600
        json.dump({
            "target_bds":[x/3 for x in bds_lst],
            "reached_bds":(bd.reshape(2)/3).tolist(),
            "dist":np.linalg.norm(np.array(bds_lst)/3-bd.reshape(2)/3),
            "num_trials":num_trials,
            "dist_std_over_trials":np.std(bd_dist_lst),
            "dist_mean_over_trials":np.mean(bd_dist_lst),
            "dist_median_over_trials":np.median(bd_dist_lst)
            },fl)

   
    print("Dist to target bd==",np.linalg.norm(np.array(bds_lst)/3-bd.reshape(2)/3))
    
    print(f"wrote to {out_fn_fig}, {out_fn_prompt_annotation_pairs}, {out_fn_bds}")


def ablate_prompt(arch):

    N=len(arch)
    for ii in range(N):

        arch[ii]._llm_descr=""



if __name__=="__main__":

    _parser = argparse.ArgumentParser(description='Train/Test the mdoel.')
    _parser.add_argument('--config', type=str,  help="yaml config", default="")

    _args=_parser.parse_args()

    with open(_args.config,"r") as fl:
        _config=yaml.load(fl,Loader=yaml.FullLoader)

    if _config["seed"]:
        _seed=_config["seed"]
        print(colored(f"setting seed to {_seed} as specified in the config file","magenta",attrs=["bold"]))
        torch.manual_seed(_seed)
        np.random.seed(_seed)
        random.seed(_seed)


    assert _config["train_model"] or _config["pretrained_model"], "You disabled training without specifying a model to test"
    
    torch.set_default_dtype(getattr(torch,_config["dtype"]))
    _device=torch.device("cpu") if not torch.cuda.is_available() else torch.device(_config["device"])

    _kmeans_lst=(lambda : (pickle.load(fl:=open(_config["model_cfg"]["kmeans"],"rb")),fl.close())[0])()
    if _config["train_model"] or _config["test_model"]:
        #load train/val/test archives and create datasets/dataloaders
        _load_archs=lambda fn: (pickle.load(f:=open(fn,"rb")),f.close())[0]
        _arch_train_path=_config["train_cfg"]["data_path_train"]
        _arch_val_path=_config["train_cfg"]["data_path_val"]
        _arch_test_path=_config["test_cfg"]["data_path"]
        _arch_train, _arch_val, _arch_test=[_load_archs(x) for x in [_arch_train_path, _arch_val_path, _arch_test_path]]

        if _config["train_cfg"]["ablate_prompt"]:
            print(colored("Ablating prompts for training and validation","red",attrs=["bold"]))
            ablate_prompt(_arch_train)
            ablate_prompt(_arch_val)
        if _config["test_cfg"]["ablate_prompt"]:
            print(colored("Ablating prompts for testing split","red",attrs=["bold"]))
            ablate_prompt(_arch_test)


        assert isinstance(_arch_train[0]._tau["action"],tuple), "please first expresset the actions using kmeans representation (see data_tools.py)"
        _cmd_dims=_arch_train[0]._tau["action"][1].shape[1]
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

        _err_msg="model info does not match config file."
        assert _model.config.block_size==_config["model_cfg"]["block_size"], _err_msg
        assert _model.config.n_head==_config["model_cfg"]["n_head"], _err_msg
        assert _model.config.n_embd==_config["model_cfg"]["n_embd"], _err_msg
        assert _model.config.n_layer==_config["model_cfg"]["n_layer"], _err_msg

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
                kmeans_obj_lst=_kmeans_lst
                )

        _model=nanoGPT_QDRL.GPT_QDRL(_gpt_cfg)

    #only has an effect if a request to change a head is made
    _model=_model.reconfigure_heads(_config["model_cfg"]["replace_head"])

    if any(_config["model_cfg"]["replace_head"].values()):
        assert _config["train_model"] or not _config["deploy_in_env"], "You are not training the model but have asked to reinitialize heads for testing/deployement. That doesn't sound wise."

    _model.to(_device)

    _input_normalizer=None
    _nav_env=None
    if any(_config["input_normalization"]["normalize"]):
        if _config["input_normalization"]["env_type"]=="navigation_env":
            from dataset_tools import make_navigation_env
            _nav_env=make_navigation_env()
            _input_normalizer=functools.partial(_nav_env.normalize_bd_obs,options=_config["input_normalization"]["normalize"],dbg=True)
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
                kmeans_lst=_kmeans_lst,
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
                kmeans_lst=_kmeans_lst,
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
        #np.random.shuffle(prompt_lst)

        policy=nanoGPT_QDRL.QDRLPolicy(
                _model,
                tokenizer=_tokenizer,
                device=_device,
                input_normalizer=_input_normalizer,
                generation_strategy=_config["deploy_cfg"]["generation_strategy"])

        scene = create_env_with_objects("./environment/")

        num_prompts=0
        basename=MiscUtils.create_directory_with_pid(_config["logging"]["log_dir"]+"/deploy_log_",remove_if_exists=True,no_pid=False)
        for prompt in prompt_lst:

            print(f"processing prompt {num_prompts}/{len(prompt_lst)}")
            generate_traj(
                    prompt,
                    num_prompts,
                    basename,
                    _nav_env.get_bd_dims(),
                    policy,
                    _nav_env,
                    scene,
                    ablate_text=_config["deploy_cfg"]["ablate_text"],
                    ablate_bd=_config["deploy_cfg"]["ablate_bd"],
                    num_trials=5)
            num_prompts+=1


