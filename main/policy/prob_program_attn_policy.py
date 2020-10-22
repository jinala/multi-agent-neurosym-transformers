import torch
import numpy as np 
import random
from sklearn import tree

from main.policy.program_attn_policy import *

dev = torch.device("cuda")



class ProbBoolCond:
    def __init__(self, probs, linear_conds):
        self.probs = probs
        self.linear_conds = linear_conds
    
    @classmethod
    def random(cls, cond_depth, num_features):
        assert(cond_depth == 2)
        num_leaves = 2 
        linear_conds = [] 
        for i in range(num_leaves):
            linear_conds.append(ProbLinearCond.random(num_features))


        num_trees = 3
        probs = torch.empty(num_trees, dtype = torch.float32).uniform_(0, 1).to(dev)

        return cls(probs, linear_conds)

    def sample_next(self, std = 0.2):
        new_probs = torch.clamp(torch.normal(self.probs, std = std).to(dev), 0, 1)
        new_linear_conds = [c.sample_next(std) for c in self.linear_conds]
        return ProbBoolCond(new_probs, new_linear_conds)


    def sample_prog(self):
        idx = torch.multinomial(self.probs + 1e-5, 1)
        if idx == 0:
            return self.linear_conds[0].sample_prog()
        else:
            l = self.linear_conds[0].sample_prog()
            r = self.linear_conds[1].sample_prog()
            if idx == 1:
                return AndCond(l, r)
            else:
                return OrCond(l, r)
          

class ProbLinearCond:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def random(cls, num_features):
        weights = torch.empty(num_features + 1, dtype = torch.float32).uniform_(-1, 1).to(dev)
        return cls(weights)

    def sample_next(self, std = 0.2):
        new_weights = torch.clamp(torch.normal(self.weights, std = std).to(dev), -1, 1)
        return ProbLinearCond(new_weights)

    def sample_prog(self):
        return LinearCond(self.weights)

class ProbAndCond:
    def __init__(self, cond1, cond2):
        self.cond1 = cond1
        self.cond2 = cond2
    
    @classmethod
    def random(cls, num_features):
        assert(False)

    def sample_next(self, std = 0.2):
        new_cond1 = self.cond1.sample_next(std = std)
        new_cond2 = self.cond2.sample_next(std = std)
        return ProbAndCond(new_cond1, new_cond2)

    def sample_prog(self):
        cond1 = self.cond1.sample_prog()
        cond2 = self.cond2.sample_prog()
        return AndCond(cond1, cond2)

class ProbOrCond:
    def __init__(self, cond1, cond2):
        self.cond1 = cond1
        self.cond2 = cond2
    
    @classmethod
    def random(cls, num_features):
        assert(False)

    def sample_next(self, std = 0.2):
        new_cond1 = self.cond1.sample_next(std = std)
        new_cond2 = self.cond2.sample_next(std = std)
        return ProbOrCond(new_cond1, new_cond2)

    def sample_prog(self):
        cond1 = self.cond1.sample_prog()
        cond2 = self.cond2.sample_prog()
        return OrCond(cond1, cond2)

class ProbLinearMap:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def random(cls, num_features):
        weights = torch.empty(num_features + 1, dtype = torch.float32).uniform_(-1, 1).to(dev)
        return cls(weights)

    def sample_next(self, std = 0.2):
        new_weights = torch.clamp(torch.normal(self.weights, std = std).to(dev), -1, 1)
        return ProbLinearMap(new_weights)

    def sample_prog(self):
        return LinearMap(self.weights)


class ProbDetAggRule():
    def __init__(self, prob_filter_cond, prob_map_fun):
        self.prob_filter_cond = prob_filter_cond
        self.prob_map_fun = prob_map_fun

    @classmethod
    def random(cls, cond_depth, num_features):
        prob_filter_cond = ProbBoolCond.random(cond_depth, num_features)
        prob_map_fun = ProbLinearMap.random(num_features)
        return cls(prob_filter_cond, prob_map_fun)

    def sample_next(self, std = 0.2):
        next_prob_filter_cond = self.prob_filter_cond.sample_next(std=std)
        next_prob_map_fun = self.prob_map_fun.sample_next(std=std)

        return ProbDetAggRule(next_prob_filter_cond, next_prob_map_fun)

    def sample_prog(self):
        filter_cond = self.prob_filter_cond.sample_prog()
        map_fun = self.prob_map_fun.sample_prog()
        return DetAggRule(filter_cond, map_fun)

class ProbNonDetRule():
    def __init__(self, prob_filter_cond):
        self.prob_filter_cond = prob_filter_cond

    @classmethod
    def random(cls, cond_depth, num_features):
        prob_filter_cond = ProbBoolCond.random(cond_depth, num_features)
        return cls(prob_filter_cond)

    def sample_next(self, std = 0.2):
        next_prob_filter_cond = self.prob_filter_cond.sample_next(std)

        return ProbNonDetRule(next_prob_filter_cond)

    def sample_prog(self):
        filter_cond = self.prob_filter_cond.sample_prog()
        return NonDetRule(filter_cond)

class ProbProgramPolicy:
    def __init__(self, probs, filters, maps, with_non_det=True):
        # probs - prob  that a rule is DetAggRule
        self.probs = probs
        self.filters = filters
        self.maps = maps
        self.with_non_det = with_non_det

    @classmethod
    def random(cls, num_rules = 2, cond_depth = 2, num_features= 12, with_non_det = True):
        probs = torch.empty(num_rules, dtype = torch.float32).uniform_(0, 1).to(dev)
        filters = []
        maps = []
        for i in range(num_rules):
            filters.append( ProbBoolCond.random(cond_depth, num_features))
            maps.append( ProbLinearMap.random(num_features) )

        return cls(probs, filters, maps, with_non_det)


    def sample_next(self, std = 0.2):
        new_probs = torch.clamp(torch.normal(self.probs, std = std).to(dev), 0, 1)
        new_filters = [r.sample_next(std=std) for r in self.filters]
        new_maps = [r.sample_next(std=std) for r in self.maps]

        return ProbProgramPolicy(new_probs, new_filters, new_maps, with_non_det = self.with_non_det)


    def sample_prog(self):
        v = torch.bernoulli(self.probs)
        rules = []
        for i in range(len(v)):
            f = self.filters[i].sample_prog()
            m = self.maps[i].sample_prog()
            if v[i] > 0.5 or (not self.with_non_det):
                rules.append(DetAggRule(f, m))
            else:
                rules.append(NonDetRule(f))
        return ProgramPolicy(rules)
        
 