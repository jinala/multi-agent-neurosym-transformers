import os
import sys
import dgl
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


class OutputManager(object):
    def __init__(self, result_path, filename='log.txt'):
        self.log_file = open(os.path.join(result_path, filename),'w')

    def say(self, s):
        self.log_file.write("{}\n".format(s))
        self.log_file.flush()
        sys.stdout.write("{}\n".format(s))
        sys.stdout.flush()


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)
    else:
        raise Exception('Result folder for this experiment already exists')

def create_dir1(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)




def dist_cartesian(p1, p2):
    return (p1 - p2).norm()


def normalize(p):
    norm = p.norm()
    if norm < 1e-6:
        # Zero vector
        return p
    else:
        return p/norm

