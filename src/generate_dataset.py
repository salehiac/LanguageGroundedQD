import functools

from deap import tools as deap_tools

import Agents
import HardMaze
import NS
import MiscUtils


if __name__=="__main__":

    assets={"env_im": "./env_assets/maze_19_2.pbm","xml_path":"./env_assets/maze_setup_5x5.xml"}
    problem=HardMaze.HardMaze(bd_type="generic",max_steps=2000, assets=assets)
    
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
            logs_root="/tmp/",
            initial_pop=[],
            problem_sampler=None)

    stop_on_reaching_task=False
    nov_estimator.log_dir=ns.log_dir_path
    ns.save_archive_to_file=True
        
    _, _=ns(iters=1000,stop_on_reaching_task=False)
    
