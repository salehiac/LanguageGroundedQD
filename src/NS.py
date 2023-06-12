import time
import os
import copy
import functools
import random
import numpy as np
import gc
import pickle
from scoop import futures

from termcolor import colored

from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

import MiscUtils


class ListArchive:

    def __init__(self, 
            max_size=200, 
            growth_rate=6,
            growth_strategy="random",
            removal_strategy="random"):
        self.max_size=max_size
        self.growth_rate=growth_rate
        self.growth_strategy=growth_strategy
        self.removal_strategy=removal_strategy
        self.container=list()

    def reset(self):
        self.container.clear()

    def update(self, parents, offspring, thresh=0, boundaries=[], knn_k=-1):
        
        pop=parents+offspring
        if self.growth_strategy=="random":
            r=random.sample(range(len(pop)),self.growth_rate)
            candidates=[pop[i] for i in r[:self.growth_rate]]
        elif self.growth_strategy=="most_novel":
            sorted_pop=sorted(pop, key=lambda x: x._nov)[::-1]#descending order
            candidates=sorted_pop[:self.growth_rate]
       
        candidates=[c for c in candidates if c._nov>thresh]
        self.container+=candidates

        if len(self)>=self.max_size:
            self.manage_size(boundaries, parents=np.concatenate([x._behavior_descr for x in parents],0).transpose(), knn_k=knn_k)

    def manage_size(self,boundaries=[],parents=[],knn_k=-1):
        if self.removal_strategy=="random":
            r=random.sample(range(len(self)),k=self.max_size)
            self.container=[self.container[i] for i in r]
        else:
            raise NotImplementedError("manag_size")

    def dump(self, fn):
        with open(fn,"wb") as f:
            pickle.dump(self.container,f)

    def __len__(self):
        return len(self.container)
    
    def __iter__(self):
        return iter(self.container)

    def __str__(self):
        return str(self.container)

class ArchiveBasedNoveltyEstimator:
    def __init__(self, k):
        self.k=k
        self.archive=None
        self.pop=None
        self.log_dir="/tmp/"

    def update(self, pop, archive):
        self.archive=archive
        self.pop=pop
 
        self.pop_bds=[x._behavior_descr for x in self.pop]
        self.pop_bds=np.concatenate(self.pop_bds, 0)
        self.archive_bds=[x._behavior_descr for x in self.archive] 
        
        if len(self.archive_bds):
            self.archive_bds=np.concatenate(self.archive_bds, 0) 
       
        self.kdt_bds=np.concatenate([self.archive_bds,self.pop_bds],0) if len(self.archive_bds) else self.pop_bds
        self.kdt = KDTree(self.kdt_bds, leaf_size=20, metric='euclidean')

    def __call__(self):
        dists, ids=self.kdt.query(self.pop_bds, self.k, return_distance=True)
        #the first column is the point itself because the population itself is included in the kdtree
        dists=dists[:,1:]
        ids=ids[:,1:]

        novs=dists.mean(1)
        return novs.tolist()

class GenericBD:
    def __init__(self, dims, num):
        self.dims=dims
        self.num=num

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a-b)

    def extract_behavior(self, trajectory):
        vec=np.zeros([self.num, self.dims])
        assert trajectory.shape[1]>=vec.shape[1], "not enough dims to extract"
        M=trajectory.shape[0]
        N=vec.shape[0]
        rem=M%N
        inds=list(range(M-1,rem-1,-(M//N)))
        vec=trajectory[inds,:self.dims]

        assert len(inds)==self.num, "wrong number of samples, this shouldn't happen"

        return vec

    def get_bd_dims(self):

        return self.dims*self.num


class NoveltySearch:

    BD_VIS_DISABLE=0
    BD_VIS_TO_FILE=1
    BD_VIS_DISPLAY=2

    def __init__(self,
            archive,
            nov_estimator,
            mutator,
            problem,
            selector,
            n_pop,
            n_offspring,
            agent_factory,
            visualise_bds_flag,
            map_type="scoop",
            logs_root="/tmp/ns_log/",
            initial_pop=[],#make sure they are passed by deepcopy
            problem_sampler=None):
        
        self.archive=archive
        if archive is not None:
            self.archive.reset()

        self.nov_estimator=nov_estimator
        self.problem=problem

        if problem_sampler is not None:
            assert self.problem is None, "you have provided a problem sampler and a problem. Please decide which you want, and set the other to None."
            self.problem=problem_sampler(num_samples=1)[0]
        
        self.map_type=map_type
        self._map=futures.map if map_type=="scoop" else map

        self.mutator=mutator
        self.selector=selector

        self.n_offspring=n_offspring
        self.agent_factory=agent_factory
       
        self.num_agent_instances=n_pop#this is important for attributing _idx values to future agents
        
        if not len(initial_pop):
            print(colored("[NS info] No initial population prior, initialising from scratch", "magenta",attrs=["bold"]))
            initial_pop=[self.agent_factory(i) for i in range(n_pop)]
            initial_pop=self.generate_new_agents(initial_pop, generation=0)
        else:
            assert len(initial_pop)==n_pop," this shouldn't happen"
            for x in initial_pop:
                x._created_at_gen=0
        

        self._initial_pop=copy.deepcopy(initial_pop)
      
        assert n_offspring>=len(initial_pop) , "n_offspring should be larger or equal to n_pop"

        self.visualise_bds_flag= visualise_bds_flag

        if os.path.isdir(logs_root):
            self.logs_root=logs_root
            self.log_dir_path=MiscUtils.create_directory_with_pid(dir_basename=logs_root+"/NS_log_"+MiscUtils.rand_string()+"_",remove_if_exists=True,no_pid=False)
            print(colored("[NS info] NS log directory was created: "+self.log_dir_path, "green",attrs=["bold"]))
        else:
            raise Exception("Root dir for logs not found. Please ensure that it exists before launching the script.")

        self.task_solvers={}#key,value=generation, list(agents)

        self.save_archive_to_file=True


    def eval_agents(self, agents):
        tt1=time.time()
        xx=list(self._map(self.problem, agents))#attention, don't deepcopy the problem instance, see Problem.py. Use problem_sampler if necessary instead.
        tt2=time.time()
        elapsed=tt2-tt1
        task_solvers=[]
        show_behaviors=False
        for ag_i in range(len(agents)):
            ag=agents[ag_i]
            ag._fitness=xx[ag_i][0]
            ag._tau=xx[ag_i][1]
            ag._behavior=xx[ag_i][2]
            ag._behavior_descr=xx[ag_i][3]
            ag._solved_task=xx[ag_i][4]

            if show_behaviors:
                self.problem.visualise_behavior(ag,hold_on=True)
            
            if hasattr(self.problem, "get_task_info"):
                ag._task_info=self.problem.get_task_info()
                
            
            if ag._solved_task:
                task_solvers.append(ag)
        
        if show_behaviors:
            plt.show()

            
        return task_solvers, elapsed



    def __call__(self, iters, stop_on_reaching_task=True, reinit=False):
        
        print(f"Starting NS with pop_sz={len(self._initial_pop)}, offspring_sz={self.n_offspring}", flush=True)

        if reinit and self.archive is not None:
            self.archive.reset()

        parents=copy.deepcopy(self._initial_pop)#pop is a member in order to avoid passing copies to workers
        self.eval_agents(parents)
       
        self.nov_estimator.update(archive=[], pop=parents)
        novs=self.nov_estimator()#computes novelty of all population
        for ag_i in range(len(parents)):
            parents[ag_i]._nov=novs[ag_i]
            

        for it in range(iters):


            offsprings=self.generate_new_agents(parents, generation=it+1)#mutations and crossover happen here  <<= deap can be useful here
            task_solvers, _ =self.eval_agents(offsprings)
           
            pop=parents+offsprings #all of them have _fitness and _behavior_descr now

            for x in pop:
                if x._age==-1:
                    x._age=it+1-x._created_at_gen
                else:
                    x._age+=1


            self.nov_estimator.update(archive=self.archive, pop=pop)
            novs=self.nov_estimator()#computes novelty of all population
            for ag_i in range(len(pop)):
                pop[ag_i]._nov=novs[ag_i]
            
            if hasattr(self.nov_estimator, "train"):
                self.nov_estimator.train(np.random.choice(pop,size=self.nov_estimator.growth_rate).tolist())

            parents_next=self.selector(individuals=pop, fit_attr="_nov")
            parents=parents_next
           
            if self.archive is not None:
                self.archive.update(parents, offsprings, thresh=self.problem.dist_thresh, boundaries=[0,600],knn_k=15)
                if self.save_archive_to_file:
                    self.archive.dump(self.log_dir_path+f"/archive_{it}")
            
            self.visualise_bds(parents + [x for x in offsprings if x._solved_task],generation_num=it)
            
            if len(task_solvers):
                print(colored("[NS info] found task solvers (generation "+str(it)+")","magenta",attrs=["bold"]),flush=True)
                self.task_solvers[it]=task_solvers
                if stop_on_reaching_task:
                    break
            gc.collect()

        return parents, self.task_solvers#iteration:list_of_agents


    def generate_new_agents(self, parents, generation:int):
       
        parents_as_list=[(x._idx, x.get_flattened_weights(), x._root) for x in parents]
        parents_to_mutate=random.choices(range(len(parents_as_list)),k=self.n_offspring)#note that usually n_offspring>=len(parents)
        mutated_genotype=[(parents_as_list[i][0], self.mutator(copy.deepcopy(parents_as_list[i][1])), parents_as_list[i][2]) for i in parents_to_mutate]#deepcopy is because of deap

        num_s=self.n_offspring if generation!=0 else len(parents_as_list)
        
        mutated_ags=[self.agent_factory(self.num_agent_instances+x) for x in range(num_s)]
        kept=random.sample(range(len(mutated_genotype)), k=num_s)
        for i in range(len(kept)):
            mutated_ags[i]._parent_idx=mutated_genotype[kept[i]][0]
            mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][1][0])
            mutated_ags[i]._created_at_gen=generation
            mutated_ags[i]._root=mutated_genotype[kept[i]][2]

        self.num_agent_instances+=len(mutated_ags)

        for x in mutated_ags:
            x.eval()
       
        return mutated_ags
    
    def visualise_bds(self, agents,generation_num=-1):
         
        if self.visualise_bds_flag!=NoveltySearch.BD_VIS_DISABLE:# and it%10==0:
            q_flag=True if self.visualise_bds_flag==NoveltySearch.BD_VIS_TO_FILE else False
            archive_it=iter(self.archive) if self.archive is not None else []
            self.problem.visualise_bds(archive_it, agents, quitely=q_flag, save_to=self.log_dir_path ,generation_num=generation_num)



