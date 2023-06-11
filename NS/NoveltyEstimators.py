from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree
import numpy as np
import torch
import pdb
import random

import matplotlib.pyplot as plt
import MiscUtils

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

