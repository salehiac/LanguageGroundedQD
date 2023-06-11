import copy
import time
import numpy as np
import cv2
import pdb
import sys
import os

from functools import reduce
import string
import random

from scoop import futures
from termcolor import colored

import gym
import gym_fastsim
import matplotlib.pyplot as plt

from NS import GenericBD 
import MiscUtils
sys.path.append("..")

class HardMaze:
    def __init__(self, bd_type="generic", max_steps=2000, display=False, assets={}):
        
        rand_str=MiscUtils.rand_string(alpha=True, numerical=False) + "-v1"
        gym_fastsim.register(id=rand_str,
                entry_point='gym_fastsim.simple_nav:SimpleNavEnv',
                kwargs={"xml_env":assets["xml_path"]})
        self.env = gym.make(rand_str)

        self.dim_obs=len(self.env.reset())
        self.dim_act=self.env.action_space.shape[0]
        self.display= display
        
        if(display):
            self.env.enable_display()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        self.bd_type=bd_type
        self.bd_extractor=GenericBD(dims=2,num=1)
        self.dist_thresh=1 #(norm, in pixels) minimum distance that a point x in the population should have to its nearest neighbour in the archive+pop
        
        self.goal_radius=42

        self.maze_im=cv2.imread(assets["env_im"]) if len(assets) else None
        self.num_saved=0

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()
    
    def get_behavior_space_boundaries(self):
        return np.array([[0,600],[0,600]])

    def __call__(self, ag):

        obs=self.env.reset()
        fitness=0
        behavior_info=[] 

        task_solved=False
        for i in range(self.max_steps):
            if self.display:
                self.env.render()
                time.sleep(0.01)
            
            action=ag(obs)
            action=action.flatten().tolist() if isinstance(action, np.ndarray) else action
            obs, reward, ended, info=self.env.step(action)
            fitness+=reward
            behavior_info.append(info["robot_pos"])
            
        #check if task solved
        dist_to_goal=np.linalg.norm(np.array(info["robot_pos"][:2])-np.array([self.env.goal.get_x(), self.env.goal.get_y()]))
        if dist_to_goal < self.goal_radius:
            task_solved=True
            ended=True
           
        bd=None
        if isinstance(self.bd_extractor, GenericBD):
            bd=self.bd_extractor.extract_behavior(np.array(behavior_info).reshape(len(behavior_info), len(behavior_info[0]))) 
        
        return fitness, bd, task_solved, None , None, None, None

    def visualise_bds(self,archive, population, quitely=True, save_to="",generation_num=-1):
        """
        currently only for 2d generic ones of size 1, so bds should be [bd_0, ...] with bd_i of length 2
        """
        if quitely and not(len(save_to)):
            raise Exception("quitely=True requires save_to to be an existing directory")
        #quitely=False

        arch_l=list(archive)
        pop_l=list(population)
        uu=arch_l+pop_l
        z=[x._behavior_descr for x in uu]
        z=np.concatenate(z,0)
        most_novel_individual_in_pop=np.argmax([x._nov for x in population])
        #pdb.set_trace()
        real_w=self.env.map.get_real_w()
        real_h=self.env.map.get_real_h()
        z[:,0]=(z[:,0]/real_w)*self.maze_im.shape[1]
        z[:,1]=(z[:,1]/real_h)*self.maze_im.shape[0]
        
        maze_im=self.maze_im.copy()

        mean_nov=np.mean([uu[i]._nov for i in range(len(uu))])

        for pt_i in range(z.shape[0]): 
            if pt_i<len(arch_l):#archive individuals
                color=MiscUtils.colors.blue
                thickness=-1
            else:#population individuals
                color=MiscUtils.colors.green
                thickness=-1
            maze_im=cv2.circle(maze_im, (int(z[pt_i,0]),int(z[pt_i,1])) , 3, color=color, thickness=thickness)
        
        maze_im=cv2.circle(maze_im,
                (int(z[len(arch_l)+most_novel_individual_in_pop,0]),int(z[len(arch_l)+most_novel_individual_in_pop,1])) , 3, color=MiscUtils.colors.red, thickness=-1)
        
        goal=self.env.map.get_goals()[0]
        
        maze_im=cv2.circle(maze_im, 
                (int(goal.get_x()*self.maze_im.shape[0]/real_h),int(goal.get_y()*self.maze_im.shape[1]/real_w)),
                3, (0,0,0), thickness=-1)
        maze_im=cv2.circle(maze_im, 
                (int(goal.get_x()*self.maze_im.shape[0]/real_h),int(goal.get_y()*self.maze_im.shape[1]/real_w)),
                int(self.goal_radius*self.maze_im.shape[0]/real_h), (0,0,0), thickness=1)

        if not quitely:
            plt.imshow(maze_im)
            plt.show()
        else:
            if len(save_to):
                b,g,r=cv2.split(maze_im)
                maze_im=cv2.merge([r,g,b])
                gen_num=generation_num if generation_num!=-1 else self.num_saved
                cv2.imwrite(save_to+"/hardmaze_2d_bd_"+str(gen_num)+".png",maze_im)
                self.num_saved+=1

