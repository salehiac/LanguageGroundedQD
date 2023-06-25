import functools
import argparse
import pickle
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import copy
from wordcloud import WordCloud
from collections import Counter 
import torch


from termcolor import colored
from scoop import futures
from deap import tools as deap_tools

import Agents
import NavigationEnv
import NS
import MiscUtils
from environment import create_env_with_objects

_assets = {
    "env_im": "./environment/ressources/maze_19_2.pbm",
    "xml_path": "./environment/maze_setup_5x5.xml"
}


#class ArchDataset


def generate_paper_dataset(logs_root="/tmp/"):
    """
    """

    problem = NavigationEnv.NavigationEnv(bd_type="generic",
                                          assets=_assets)

    nov_estimator = NS.ArchiveBasedNoveltyEstimator(k=15)
    arch = NS.ListArchive(max_size=10000,
                          growth_rate=6,
                          growth_strategy="random",
                          removal_strategy="random")

    selector = functools.partial(MiscUtils.selBest,
                                 k=50,
                                 automatic_threshold=False)

    in_dims = problem.dim_obs
    out_dims = problem.dim_act
    num_pop = 50

    normalise_output_with = ""
    num_hidden = 3
    hidden_dim = 10

    def make_ag(idx):
        return Agents.SmallFC_FW(idx,
                                 in_d=in_dims,
                                 out_d=out_dims,
                                 num_hidden=num_hidden,
                                 hidden_dim=hidden_dim,
                                 output_normalisation=normalise_output_with)

    eta = 10
    low = -1.0
    up = 1.0
    indpb = 0.1

    mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                eta=eta,
                                low=low,
                                up=up,
                                indpb=indpb)

    ns = NS.NoveltySearch(archive=arch,
                          nov_estimator=nov_estimator,
                          mutator=mutator,
                          problem=problem,
                          selector=selector,
                          n_pop=num_pop,
                          n_offspring=50,
                          agent_factory=make_ag,
                          visualise_bds_flag=True,
                          map_type="scoop",
                          logs_root=logs_root,
                          initial_pop=[],
                          problem_sampler=None)

    stop_on_reaching_task = False
    nov_estimator.log_dir = ns.log_dir_path
    ns.save_archive_to_file = True

    _, _ = ns(iters=1000, stop_on_reaching_task=False,save_frequency=10)


def verify_repeatability_individual(ag):
    """
    return True if the policy's info is repeatable
    """

    problem = NavigationEnv.NavigationEnv(bd_type="generic",
                                          assets=_assets)

    fitness, tau, behavior, bd, task_solved = problem(ag)

    attrs = [
        "_fitness", "_tau", "_behavior", "_behavior_descr", "_solved_task"
    ]

    def check_equality(xx,yy):
        if isinstance(xx,np.ndarray):
            return (xx==yy).all()
        if isinstance(xx,float) or isinstance(xx,bool):
            return xx==yy
        if isinstance(xx,dict):
            aa=(xx["obs"]==yy["obs"]).all()
            bb=(xx["action"]==yy["action"]).all()
            return aa and bb
    
    comp = list(
        map(
            lambda z: check_equality(z[0],z[1]),
            zip([fitness, tau, behavior, bd, task_solved],
                [getattr(ag, attrs[i]) for i in range(5)])))

    return sum(comp) == len(comp)

def find_duplicates(arch):

    dupes=[]
    for ii in range(len(arch)):
        for jj in range(ii+1,len(arch)):

            if arch[ii]==arch[jj]:
                dupes.append((ii,jj))

    return dupes

if __name__ == "__main__":

    _parser = argparse.ArgumentParser(
        description=
        'Tool to generate/annotate datasets for the predefined navigation env.'
    )
    _parser.add_argument(
        '--generate_archive',
        action='store_true',
        help=
        "generate and archive of policies/trajectories/behaviors (results will be saved to --out_dir). NOTE: this should be launched with scoop, e.g. python3 -m scoop -n 32 <this_script>"
    )
    _parser.add_argument('--input_archive',
                         type=str,
                         default="",
                         help="archive to process")
    #those are only relevant if an input archive is given
    _parser.add_argument(
        '--annotate_archive',
        action='store_true',
        help=
        "annotate input archive with semantic info (pre-LLM), results will be saved to a new archive in --out_dir"
    )
    _parser.add_argument(
        '--verify_repeatability',
        action='store_true',
        help=
        "verify that the policies in the input archive produce the {(obs,action)} and behaviors that are in the archive."
    )
    _parser.add_argument(
        '--verify_repeatability_individual',
        type=int,
        default=-1,
        help="verify individual at this index instead of entire archive")
    _parser.add_argument(
        '--plot_behavior_space_traj',
        action='store_true',
        help=
        "if given, a figure of each individual's behavior is saved to --out_dir"
    )

    _parser.add_argument(
        '--export_annotations',
        action='store_true',
        help=
        "Will throw an exception if the input archive has not already been annotated. Exports annotation data to --out_dir"
    )
    _parser.add_argument(
            '--add_llm_descriptions',
            type=str,
            default="",
            help="path to a directory with files <filename>_<traj_idx>.json, each containing a dictionary with key 'descr', containing an llm generated description for traj <traj_id>. This argument will add that description to the corresponding policies, and save the archive to --out_dir.")


    _parser.add_argument(
            '--visualize_llm_description_and_behavior',
            type=int,
            default=-1,
            help="index of individual for whose behavior space trajectory we want to visualize alongside the llm generated description")

    _parser.add_argument(
            '--archive_info',
            action='store_true',
            help="prints info about --input_archive"
            )

    _parser.add_argument(
            '--fix_duplicates',
            action='store_true',
            help="Some policies can appear multiple times in the archive. We separate them as a form of data augmentation: this pairs the same policy with different but similar texts"
            )

    _parser.add_argument(
            '--split_archive',
            metavar="x",
            nargs=3,#the three arguments should be in [0,1] and sum to 1.0
            type=float,
            help="splits archive into train/val/test splits. The three input values should be normalized percentages. For example, values of [0.9, 0.05, 0.05] will keep 90%, 5%, 5% of the data as respectively train, val and test splits. The results are saved to --out_dir"
            )



    #output arg
    _parser.add_argument('--out_dir',
                         type=str,
                         default="/tmp/logdir/",
                         help="directory to save results")

    _args = _parser.parse_args()

    if len(sys.argv) == 1:
        _parser.print_help()

    if _args.generate_archive:
        generate_paper_dataset(_args.out_dir)
    if _args.input_archive:
        with open(_args.input_archive, "rb") as fl:
            _in_arch = pickle.load(fl)

        if _args.verify_repeatability:
            if int(_args.verify_repeatability_individual) != -1:
                passed = verify_repeatability_individual(_in_arch[int(
                    _args.verify_repeatability_individual)])
                print(
                    colored(
                        f"Individual's behavior was consistent with archive's content: {passed}",
                        "red" if not passed else "green",
                        attrs=["bold"]))
            else:
                passed_lst = list(
                    futures.map(lambda x: verify_repeatability_individual(x),
                                _in_arch))
                print(
                    colored(
                        f"consistent individuals: {sum(passed_lst)}/{len(_in_arch)}",
                        "blue",
                        attrs=["bold"]))

        elif int(_args.verify_repeatability_individual) != -1:
            raise Exception(
                "You have specified verify_repeatability_individual without passing the verify_repeatability flag"
            )

        if _args.plot_behavior_space_traj:

            scene = create_env_with_objects("./environment/")
            list(
                futures.map(
                    lambda x: scene.display(display_bbox=False,
                                            hold_on=False,
                                            path2d_info=(x[0]._behavior, 600, 600),
                                            save_to=f"{_args.out_dir}/{x[1]}"),
                    zip(_in_arch,
                        [f"behavior_{i}" for i in range(len(_in_arch))])))

        if _args.export_annotations:

            if not hasattr(_in_arch[0],"_annotation"):
                raise Exception("--export_annotations can only be used with an annotated archive. Did you mean to pass --annotate_archive?")

            for ag_i in range(len(_in_arch)):
                fn=f"{_args.out_dir}/annotation_{ag_i}.json"
                with open(fn,"w") as fl:
                    json.dump(_in_arch[ag_i]._annotation,fl)

        if _args.add_llm_descriptions:

            in_fns={MiscUtils.get_trailing_number(x[:-5]):_args.add_llm_descriptions+"/"+x for x in os.listdir(_args.add_llm_descriptions)}
            for ag_i in range(len(_in_arch)):
                if ag_i in in_fns:
                    print(colored(f"adding description to agent {ag_i}.","green",attrs=["bold"]))
                    with open(in_fns[ag_i],"r") as fl:
                        descr_string=json.load(fl)["descr"]
                    _in_arch[ag_i]._llm_descr=descr_string
                else:
                    pass
                    #print(colored(f"no description for agent {ag_i}, skipping it.","red",attrs=["bold"]))

            mm=[True if (hasattr(_in_arch[ii],"_llm_descr") and _in_arch[ii]._llm_descr is not None) else False for ii in range(len(_in_arch))] 
            if False in mm:
                first_non_annotated=mm.index(False)
                assert sum(mm[:first_non_annotated])==len(mm[:first_non_annotated]), "the llm descr process is incremental w.r.t policy indexes"
                assert sum(mm[first_non_annotated:])==0, "the llm descr process is incremental w.r.t policy indexes"

            out_fn=f"{_args.out_dir}/described_archive.pickle"
            print(f"saving to {out_fn}")
            with open(out_fn,"wb") as fl:
                pickle.dump(_in_arch,fl)
        
        if _args.annotate_archive:
            scene = create_env_with_objects("./environment/")

            _ag_counter = 0
            for ag in _in_arch:

                print(f"annotating trajectory of agent {_ag_counter}/{len(_in_arch)}")
                ag._annotation = scene.annotate_traj(
                    ag._behavior, real_w=600, real_h=600, step=40
                )  #note that we annotate the behavior space trajectory, not tau

                #fig,_=scene.display(display_bbox=False,hold_on=True,path2d_info=(path2d,600,600))
                #plt.show()

                _ag_counter += 1

            _annotated_archive_path = f"{_args.out_dir}/annotated_archive.pickle"
            with open(f"{_annotated_archive_path}", "wb") as fl:
                pickle.dump(_in_arch, fl)
            print(
                colored(
                    f"annotated archive was saved to {_annotated_archive_path}",
                    "green",
                    attrs=["bold"]))

        if _args.visualize_llm_description_and_behavior!=-1:

            _idx=_args.visualize_llm_description_and_behavior
            scene = create_env_with_objects("./environment/")
            
            if not hasattr(_in_arch[_idx],"_llm_descr") or _in_arch[_idx]._llm_descr is None :
                raise Exception(colored(f"Element {_idx} has not been described. See the input options for annotation archives and adding LLM descriptions","red",attrs=["bold"]))

            fig,_=scene.display(display_bbox=False,
                    hold_on=True,
                    path2d_info=(_in_arch[_idx]._behavior, 600, 600))

            fig.suptitle(MiscUtils.add_newlines(_in_arch[_idx]._llm_descr))
            plt.tight_layout()
            plt.show()

        if _args.archive_info:

           import pprint
           _sz=len(_in_arch)
           _dd={}
           _dd["size"]=_sz
           _dd["num annotated"]=functools.reduce(lambda acc, x: acc+1 if (hasattr(x,"_annotation") and x._annotation is not None) else acc, _in_arch,0)
           _dd["num described"]=functools.reduce(lambda acc, x: acc+1 if (hasattr(x,"_llm_descr") and x._llm_descr is not None) else acc, _in_arch,0)

           mm=[True if (hasattr(x,"_llm_descr") and x._llm_descr is not None) else False for x in _in_arch] 
           if False in mm:
               first_non_annotated=mm.index(False)
               assert sum(mm[:first_non_annotated])==len(mm[:first_non_annotated]), "the llm descr process is incremental w.r.t policy indexes"
               assert sum(mm[first_non_annotated:])==0, "the llm descr process is incremental w.r.t policy indexes"
           else:
               first_non_annotated=None

           _dd["described from/to"]=f"[0,{first_non_annotated})" if first_non_annotated is not None else "fully described"
           _dd["episode length"]=_in_arch[0]._tau["action"].shape[0]
           _dd["cmd dims"]=_in_arch[0]._tau["action"].shape[1]
           _dd["obs dims"]=_in_arch[0]._tau["obs"].shape[1]
           _dd["bd dims"]=_in_arch[0]._behavior_descr.shape[1]

           from tokenizers import pre_tokenizers
           pre_tok=pre_tokenizers.Whitespace()

           word_lst=[]
           for _ag in _in_arch:
               word_lst+=[x[0] for x in pre_tok.pre_tokenize_str(_ag._llm_descr.lower())]
           word_lst_unique=list(set(word_lst))

           _dd["num unique words (pretokens)"]=len(word_lst_unique)

           cnt=Counter(word_lst)
           zz=sorted([(k,v) for k,v in cnt.items()],key=lambda x: x[1],reverse=True) 
           _thresh=3
           _dd[f"num pretokens appearing more than {_thresh} times"]=len([x for x in zz if x[1]>_thresh])

           
           _pp = pprint.PrettyPrinter(indent=4,sort_dicts=False)
           _pp.pprint(_dd)

           import warnings
           warnings.filterwarnings("ignore")
           word_dict = dict(zz)
           wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_dict)
           wc_path='/tmp/wordcloud.png'
           wordcloud.to_file(wc_path)
           print(colored(f"word cloud saved to {wc_path}","cyan",attrs=["bold"]))


        if _args.fix_duplicates:

            #_dupes=find_duplicates(_in_arch)
            #print(_dupes)

            _in_arch_cp=[copy.deepcopy(x) for x in _in_arch]

            with open(f"{_args.out_dir}/archive_fixed_duplicates.pkl","wb") as fl:
                pickle.dump(_in_arch_cp,fl)

        if _args.split_archive:

            np.random.shuffle(_in_arch)
            _r_train=_args.split_archive[0]
            _r_val=_args.split_archive[1]
            _r_test=_args.split_archive[2]

            assert _r_train+_r_val+_r_test==1.0

            _num=len(_in_arch)
            _idx_1=int(_r_train*_num)
            _idx_2=_idx_1+int(_r_val*_num)+1

            _train_arch=_in_arch[:_idx_1]
            _val_arch=_in_arch[_idx_1:_idx_2]
            _test_arch=_in_arch[_idx_2:]

            basename=_args.input_archive.split("/")[-1].split(".")[0]
            out_f_train=f"{_args.out_dir}/{basename}_train.pkl"
            out_f_val=f"{_args.out_dir}/{basename}_val.pkl"
            out_f_test=f"{_args.out_dir}/{basename}_test.pkl"
            
            def dump_arch(arch, fn, msg=""):
                with open(fn,"wb") as fl:
                    pickle.dump(arch,fl)
                if msg:
                    print(msg)

            dump_arch(_train_arch, out_f_train, msg=colored(f"wrote train archive to {out_f_train}","cyan",attrs=["bold"]))
            dump_arch(_val_arch, out_f_val, msg=colored(f"wrote val archive to {out_f_val}","cyan",attrs=["bold"]))
            dump_arch(_test_arch, out_f_test, msg=colored(f"wrote test archive to {out_f_test}","cyan",attrs=["bold"]))




