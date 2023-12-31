
# Integrating LLMs and Decision Transformers for Language Grounded Generative Quality-Diversity

This is the official implementation of the method proposed in 

```
Integrating LLMs and Decision Transformers for Language Grounded Generative Quality-Diversity
Salehi, A., Doncieux, S. (2023).
arXiv preprint https://arxiv.org/abs/2308.13278
```

**Note 1:** This repository requires `python3.8+` and `pytorch<2.0`. The results in the paper were produced with pytorch `1.11.0+cu113`. 

**Note 2:** The transformer included in this repo builds on the nanoGPT (https://github.com/karpathy/nanoGPT) language model, which it extends to the conditional RL/QD case.

## Prerequisites

### OpenAI API key

You need to add your openAI API key to the script `get_trajectory_description.py` (`openai.api_key="ADD YOUR KEY HERE"`). If you want to evalute trajectories generated by an agent, do the same for the `eval_trajectory.py` script.

### Toy environment dependencies

1. Libfastsim (<https://github.com/sferes2/libfastsim>)
2. pyfastsim  (<https://github.com/mirandablue/pyfastsim>)
3. fastsim_gym (<https://github.com/mirandablue/fastsim_gym>)

## Generating datasets 

Run the following commands, *using the archive generated at each step* as input to the next:
```
cd src
python3 -m scoop -n <num_processes> dataset_tools.py --generate_archive --out_dir <some_path>  #this will use NoveltySearch to generate an archive of policies. A good rule of thumb is to set num_processes to approximately the number of cores. 
python3 dataset_tools.py --fix_duplicates  --input_archive <path/to/generated_archive> --out_dir <some_path> 
python3 dataset_tools.py --input_archive <path/to/fixed_archive> --annotate_archive --out_dir <some_path>
python3 dataset_tools.py --export_annotations --input_archive <path/to/annotated_archive> --out_dir <annotations_dir>
```

Now use the `get_trajectory_description` script to generate descriptions for the annotations that have been written to `annotations_dir`:

```
python3 get_trajectory_description <annotations_dir> <description_dir> #the output is written to description_dir
```

Finally, add the llm generated descriptions to the archive:

```
python3 dataset_tools.py --input_archive <path/to/annotated_archive> --add_llm_descriptions <description_dir> --out_dir <some_path> 
```

It is recommended that you generate several archives (preferably with different values for `NavigationEnv.dist_thresh`), that you can then combine using `python3 dataset_tools --merge_archives`. Once you have a sufficiently large dataset, you can call 

```
python3 dataset_tools --prepare_actions_for_multimodal <num_dim_0_clusters> <num_dim_1_clusters> --out_dir <some_path> 
```

to perform kmeans clustering and express the actions as `cluster_center+offset`, as specified in the paper. This will write two files, `cluster_centers` and `transformed_archive.pkl` to the path given to `--out_dir`. The transformed archive can now be divided into train/val/test splits via `--split_archive`, *e.g.* 

```
python3 dataset_tools --split_archive  0.8 0.1 0.1 #80%,10%,10% of the archive used for train, val, test splits
```
Note that splitting the archive shuffles the order of the elements. The `cluster_centers` file will be used during training as documented below.

## Training the model

First, please edit the config file `src/config/config.yaml` by replacing the `<path_to_*>` entries with appropriate paths. Note that a trained tokenizer is provided in `src/utils/tokenizers/`, whose path can be specified under the config file's `learned_tokenizer` entry. Alternatively, you can train a tokenizer on your archive's corpus using the `src/TokenizingTools.py` script. Make sure the `train_model` flag is set to True. Note that `["model_cfg"]["kmeans"]` should be set to the path of the `cluster_centers` file that was obtained at the end of the previous section. Once the file is edited, call

```
cd src
python3 LanguageConditionedQDRLTransformer.py --config config/config.yaml
```

to train the transformer. Note that you can pass an empty string to `["depoly_cfg"]["prompts"]` in the config file, as the `depoly_cfg` section is not relevant for training.

The training process will create a logging directory `["logging"]["log_dir"]/train_val_log_{PID_or_random_str}` with a randomized name, where various statistics such as loss values (in *json format) as well as model checkpoints will be saved during training.

## Evaluation

For evaluation, we first deploy the transformer into the environment given different behavior descriptor and prompt conditionings, and then use an LLM via the `eval_trajectory.py` script to rate the trajectories.

To deploy the policiy in test environments, set the `deploy_in_env` flag to True in the config file. There is no need to set the `test_model` flag to True as the latter will only compute the loss function on the test split, and this is of little interest here. Now run 

```
python3 dataset_tools.py --input_archive <archive_test_split.pkl> --export_as_prompt_list --out_dir <your_output_dir>
```

This will create a file `<your_output_dir>/prompt_list.json`, whose path should then be given under the `["deploy_cfg"]["prompts"]` entry in the config file. Setting `deploy_in_env` to True in the config file, 

```
cd src
python3 LanguageConditionedQDRLTransformer.py --config config/config.yaml
```
will run the transformer in the enviromnent for each test in the `prompt_list.json` file, annotate the trajectory with semantic information, and save the results in the path specified under `logging` in the config file. Once this is done, those results can be used as input to the `src/eval_trajectory.py` script, using the `--llm_eval` flag. Note that passing the flag `--human_eval` will open up a GUI for each trajectory which will allow a human user to assign a score to a trajectory/prompt coupling, which can be useful if one wants to compute correlation/similarity scores between human and LLM based trajectory evaluations.

# Additional notes

The images used in the toy environment to indicate household objects are vector images under creative commons licence, downloaded from openclipart.org.
