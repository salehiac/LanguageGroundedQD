import functools
import argparse
import pickle
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

from termcolor import colored
from scoop import futures
from deap import tools as deap_tools

import Agents
import NavigationEnv 
import NS
import MiscUtils
from environment import create_env_with_objects

_assets={"env_im": "./environment/ressources/maze_19_2.pbm","xml_path":"./environment/maze_setup_5x5.xml"}

def generate_paper_dataset(logs_root="/tmp/"):
    """
    """

    problem=NavigationEnv.NavigationEnv(bd_type="generic",max_steps=2000, assets=_assets)
    
    nov_estimator= NS.ArchiveBasedNoveltyEstimator(k=15)
    arch=NS.ListArchive(max_size=10000,
            growth_rate=6,
            growth_strategy="random",
            removal_strategy="random")

    selector=functools.partial(MiscUtils.selBest,k=50,automatic_threshold=False)

    in_dims=problem.dim_obs
    out_dims=problem.dim_act
    num_pop=50
            
    normalise_output_with="" 
    num_hidden=3
    hidden_dim=10

    def make_ag(idx):
        return Agents.SmallFC_FW(
                        idx,
                        in_d=in_dims,
                        out_d=out_dims,
                        num_hidden=num_hidden,
                        hidden_dim=hidden_dim,
                        output_normalisation=normalise_output_with)
    
    eta=10
    low=-1.0
    up=1.0
    indpb = 0.1

    mutator=functools.partial(deap_tools.mutPolynomialBounded,eta=eta, low=low, up=up, indpb=indpb)

    ns=NS.NoveltySearch(archive=arch,
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

    stop_on_reaching_task=False
    nov_estimator.log_dir=ns.log_dir_path
    ns.save_archive_to_file=True
        
    _, _=ns(iters=1000,stop_on_reaching_task=False)

def verify_repeatability_individual(ag):
    """
    return True if the policy's info is repeatable
    """

    problem=NavigationEnv.NavigationEnv(bd_type="generic",max_steps=2000, assets=_assets)

    fitness, tau, behavior, bd, task_solved=problem(ag)

    attrs=["_fitness", "_tau", "_behavior", "_behavior_descr", "_solved_task"]

    comp=list(map(lambda x:x[0]==x[1] if not isinstance(x[0],np.ndarray) else (x[0]==x[1]).all(), zip([fitness, tau, behavior, bd, task_solved], [getattr(ag,attrs[i]) for i in range(5)])))

    return sum(comp)==len(comp)
   
    
if __name__=="__main__":


    _parser = argparse.ArgumentParser(description='Tool to generate/annotate datasets for the predefined navigation env.')
    _parser.add_argument('--generate_archive',action='store_true',help="generate and archive of policies/trajectories/behaviors (results will be saved to --out_dir). NOTE: this should be launched with scoop, e.g. python3 -m scoop -n 32 <this_script>")
    _parser.add_argument('--input_archive',type=str,default="",help="archive to process")
    #those are only relevant if an input archive is given
    _parser.add_argument('--annotate_archive', action='store_true',help="annotate input archive with semantic info (pre-LLM), results will be saved to a new archive in --out_dir")
    _parser.add_argument('--verify_repeatability', action='store_true',help="verify that the policies in the input archive produce the {(obs,action)} and behaviors that are in the archive.")
    _parser.add_argument('--verify_repeatability_individual', type=int,default=-1,help="verify individual at this index instead of entire archive")
    _parser.add_argument('--plot_behavior_space_traj', action='store_true',help="if given, a figure of each individual's behavior is saved to --out_dir")
    #output arg
    _parser.add_argument('--out_dir', type=str,default="/tmp/logdir/",help="directory to save results")

    _args=_parser.parse_args()

    if len(sys.argv)==1:
        _parser.print_help()
   
    if _args.generate_archive:
        generate_paper_dataset(_args.out_dir)
    if _args.input_archive:
        with open(_args.input_archive,"rb") as fl:
            _in_arch=pickle.load(fl)
        
        if _args.verify_repeatability:
            if int(_args.verify_repeatability_individual)!=-1:
                passed=verify_repeatability_individual(_in_arch[int(_args.verify_repeatability_individual)])
                print(colored(f"Individual's behavior was consistent with archive's content: {passed}", "red" if not passed else "green",attrs=["bold"]))
            else:
                passed_lst=list(futures.map(lambda x:verify_repeatability_individual(x), _in_arch))
                print(colored(f"consistent individuals: {sum(passed_lst)}/{len(_in_arch)}","blue",attrs=["bold"]))
                
        elif int(_args.verify_repeatability_individual)!=-1:
            raise Exception("You have specified verify_repeatability_individual without passing the verify_repeatability flag")

        if _args.plot_behavior_space_traj:
            
            #_problem=NavigationEnv.NavigationEnv(bd_type="generic",max_steps=2000, assets=_assets)
            #list(futures.map(lambda x:_problem.visualise_behavior(x[0],hold_on=False,save_im_to=f"{_args.out_dir}/{x[1]}"),zip(_in_arch, [f"behavior_{i}" for i in range(len(_in_arch))])))

            scene=create_env_with_objects("./environment/")
            list(futures.map(lambda x:scene.display(display_bbox=False,hold_on=False,path2d_info=(np.stack([y[:2] for y in x[0]._behavior]),600,600),save_to=f"{_args.out_dir}/{x[1]}"),zip(_in_arch, [f"behavior_{i}" for i in range(len(_in_arch))])))
        
        if _args.annotate_archive:
            scene=create_env_with_objects("./environment/")

            _ag_counter=0
            for ag in _in_arch:
               
                print(f"annotating trajectory of agent {_ag_counter}")
                path2d=np.stack([x[:2] for x in ag._behavior])
                ag._annotation=scene.annotate_traj(path2d,real_w=600,real_h=600,step=200)#note that we annotate the behavior space trajectory, not tau
                
                #fig,_=scene.display(display_bbox=False,hold_on=True,path2d_info=(path2d,600,600))
                #plt.show()

                _ag_counter+=1

            _annotated_archive_path=f"{_args.out_dir}/annotated_archive.pickle"
            with open(f"{_annotated_archive_path}","wb") as fl:
                pickle.dump(_in_arch,fl)
            print(colored(f"annotated archive was saved to {_annotated_archive_path}","green",attrs=["bold"]))

            
                


