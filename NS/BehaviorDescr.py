from abc import ABC, abstractmethod
import numpy as np
import torch

import pdb

import MiscUtils

class BehaviorDescr:
    @staticmethod
    def distance(a, b):
        pass
    @abstractmethod
    def extract_behavior(self, x):
        pass

    @abstractmethod
    def get_bd_dims(self):
        pass

class GenericBD(BehaviorDescr):
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

