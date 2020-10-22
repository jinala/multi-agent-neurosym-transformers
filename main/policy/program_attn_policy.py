import torch
import numpy as np
import random
from sklearn import tree


dev = torch.device("cuda")

class Rule:
    def __init__():
        pass

    def eval(self, input):
        pass

class BoolCond:
    def __init__():
        pass
    def eval(self, input):
        pass

class LinearCond(BoolCond):
    def __init__(self, weights, ignore_self = True):
        self.weights = weights.to(dev)
        self.ignore_self = ignore_self

    def eval(self, input):
        # input (S, N, N, d_f)
        # output (S, N, N) with values {0,1}
        S, N, _, d_f = input.shape
        assert(len(self.weights) == d_f + 1)
        c = torch.ones(S*N*N, dtype = torch.float32).to(dev).reshape((S, N, N, 1))
        input_c = torch.cat((input, c), -1) #(S, N, N, d_f + 1)
        v = torch.tensordot(input_c, self.weights, dims=1) #(S, N, N)

        res = torch.zeros((S,N, N), dtype=torch.float32).to(dev)
        res[v >= 0] = 1
        if self.ignore_self:
            Y, X = np.meshgrid(np.arange(N), np.arange(S))
            X = X.reshape(S*N)
            Y = Y.reshape(S*N)
            res[X, Y, Y] = 0
        return res

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,) with values {0,1}
        N, d_f = input.shape
        c = torch.ones((N, 1), dtype=torch.float32).to(dev)
        input_c = torch.cat((input, c), -1) # (N, d_f + 1)
        v = torch.tensordot(input_c, self.weights, dims=1)
        res = torch.zeros((N), dtype=torch.float32).to(dev)
        res[v >= 0] = 1
        return res

    def __repr__(self):
        return "LinearCond(%s)"%repr(self.weights.cpu())

    def __str__(self):
        return weights_to_str(self.weights) + " >= 0"

class AndCond(BoolCond):
    def __init__(self, cond1, cond2):
        self.cond1 = cond1
        self.cond2 = cond2

    def eval(self, input):
        # input (S, N, N, d_f)
        # output (S, N, N) with values {0,1}
        res1 = self.cond1.eval(input)
        res2 = self.cond2.eval(input)
        return res1*res2

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,) with values {0,1}
        res1 = self.cond1.eval_separate(input)
        res2 = self.cond2.eval_separate(input)
        return res1*res2

    def __repr__(self):
        return "AndCond(%s, %s)"%(repr(self.cond1), repr(self.cond2))

    def __str__(self):
        return str(self.cond1) + " and " + str(self.cond2)

class OrCond(BoolCond):
    def __init__(self, cond1, cond2):
        self.cond1 = cond1
        self.cond2 = cond2

    def eval(self, input):
        # input (S, N, N, d_f)
        # output (S, N, N) with values {0,1}
        res1 = self.cond1.eval(input)
        res2 = self.cond2.eval(input)
        res = res1+res2
        res[res > 1e-2] = 1
        return res

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,) with values {0,1}
        res1 = self.cond1.eval_separate(input)
        res2 = self.cond2.eval_separate(input)
        res = res1+res2
        res[res > 1e-2] = 1
        return res

    def __repr__(self):
        return "OrCond(%s, %s)"%(repr(self.cond1), repr(self.cond2))

    def __str__(self):
        return str(self.cond1) + " or " + str(self.cond2)

class LinearMap:
    def __init__(self, weights):
        self.weights = weights.to(dev)

    def eval(self, input):
        # input (S, N, N, d_f)
        # output (S, N, N) with values {0,1}
        S, N, _, d_f = input.shape
        assert(len(self.weights) == d_f + 1)
        c = torch.ones(S*N*N, dtype=torch.float32).to(dev).reshape((S, N, N, 1))
        input_c = torch.cat((input, c), -1) #(S, N, N, d_f + 1)
        v = torch.tensordot(input_c, self.weights, dims=1) #(S, N, N)
        return v

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,)
        N, d_f = input.shape
        c = torch.ones((N, 1), dtype=torch.float32).to(dev)
        input_c = torch.cat((input, c), -1) # (N, d_f + 1)
        v = torch.tensordot(input_c, self.weights, dims=1)
        return v

    def __repr__(self):
        return "LinearMap(%s)"%repr(self.weights.cpu())

    def __str__(self):
        return weights_to_str(self.weights)


class DetAggRule(Rule):
    def __init__(self, filter_cond, map_fun):
        self.filter_cond = filter_cond
        self.map_fun = map_fun
        self.inhibit_self = False

    def set_no_self(self):
        self.inhibit_self = True

    def eval(self, input):
        # input (S, N, N, d_f)
        # output attn (S, N, N) with values {0,1}
        S, N, _, _ = input.shape
        fv = self.filter_cond.eval(input) #(S, N, N) boolean
        if self.inhibit_self:
            mask = torch.eye(N, dtype=bool).to(dev).unsqueeze(0)
            fv.masked_fill_(mask, 0)

        mv = self.map_fun.eval(input) # (S, N, N)
        mv[fv < 1e-2] = float('-inf')
        idx = torch.argmax(mv, dim=2) # (S, N)
        idx = idx.view(S*N) # (S*N)

        Y, X = np.meshgrid(np.arange(N), np.arange(S))
        X = X.reshape(S*N)
        Y = Y.reshape(S*N)

        attn = torch.zeros((S, N, N), dtype = torch.float32).to(dev)
        attn[X, Y, idx] = 1
        attn[mv < -1e20] = 0
        return attn

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,)
        N, d_f = input.shape
        fv = self.filter_cond.eval_separate(input)
        mv = self.map_fun.eval_separate(input)
        mv[fv < 1e-2] = -4

        return mv

    def __repr__(self):
        return "DetAggRule(%s, %s)"%(repr(self.filter_cond), repr(self.map_fun))

    def __str__(self):
        return "DetAggRule(%s, %s)"%(str(self.filter_cond), str(self.map_fun))


class NonDetRule(Rule):
    def __init__(self, filter_cond):
        self.filter_cond = filter_cond
        self.inhibit_self = False

    def set_no_self(self):
        self.inhibit_self = True

    def eval(self, input):
        # input (S, N, N, d_f)
        # output attn (S, N, N) with values {0,1}
        S, N, _, _ = input.shape
        fv = self.filter_cond.eval(input) #(S, N, N) boolean

        #s = fv.sum(dim = -1).unsqueeze(2) + 1e-4 #(S, N, N)
        #p = fv/s

        #return torch.bernoulli(p)
        if self.inhibit_self:
            mask = torch.eye(N, dtype=bool).to(dev).unsqueeze(0)
            fv.masked_fill_(mask, 0)

        fv = fv.view(S*N, N) + 1e-10
        attn = torch.zeros(S*N, N, dtype = torch.float32).to(dev)
        idx = torch.multinomial(fv, 1)
        attn[np.arange(S*N), idx.view(-1)] = 1
        attn[fv < 1e-2] = 0

        return attn.reshape(S, N, N)

    def eval_separate(self, input):
        # input (N, d_f)
        # output (N,), {0, 1}
        N, d_f = input.shape
        fv = self.filter_cond.eval_separate(input)

        return fv

    def __repr__(self):
        return "NonDetRule(%s)"%repr(self.filter_cond)

    def __str__(self):
        return "NonDetRule(%s)"%str(self.filter_cond)

class ProgramPolicy:
    def __init__(self, rules):
        self.rules = rules
        self.dt_self_edge = None
        self.only_obs_edge = False

    def set_dt_self_edge(self, dt_se):
        self.dt_self_edge = dt_se

    def set_no_self(self):
        for rule in self.rules:
            rule.set_no_self()

    def set_only_edge(self):
        self.only_obs_edge = True

    def eval(self, X):
        # input (S, N, N , d), S is batch size, N is number of robots
        # edge_input (S, N^2, d_in_edge)

        S, N, _, _ = X.shape

        attn = torch.zeros((S, N, N), dtype = torch.float32).to(dev)
        for rule in self.rules:
            attn += rule.eval(X)

        # calculate self attn
        if self.dt_self_edge != None:
            inp = input.view(S*N, d_in).data.cpu().numpy()
            out = self.dt_self_edge.predict(inp)
            out = torch.tensor(out, dtype=torch.float32).to(dev)
            self_attn = torch.zeros((S, N, N), dtype = torch.float32).to(dev)

            Y, X = np.meshgrid(np.arange(N), np.arange(S))
            X = X.reshape(N*S)
            Y = Y.reshape(N*S)
            self_attn[X, Y, Y] = out

            attn += self_attn

        attn[attn > 1e-2] = 1
        sums = attn.sum(axis = -1) + 1e-4
        sums = sums.reshape((S, N, 1))
        attn = attn/sums

        return attn

    def __repr__(self):
        return "ProgramPolicy(%s)"%repr(self.rules).replace("tensor", "torch.tensor").replace("\n", " ")

    def __str__(self):
        return str([str(r) for r in self.rules])

def weights_to_str(weights):
    names = ["my_goal_x/20", "my_goal_y/20", "my_goal_d/25", "my_goal_ang/3.14",  "rel_x/20", "rel_y/20", "rel_d/25", "rel_ang/3.14"]
    #return str(weights)
    indices = weights.nonzero()

    res = ""
    for i in range(len(indices)):
        idx = indices[i]
        w = weights[idx]

        if i != 0 and w > 0:
            res +=  " + "
        elif w < 0:
            res += " - "

        if abs(w) != 1:
            res += str(abs(w.data.cpu().numpy()[0]))

        if idx < len(names):
            res += names[idx]
    return res

def sample_prog_policy(num_rules = 2, cond_depth = 2, num_features = 12):
    rules = []
    for i in range(num_rules):
        type = random.randint(0, 1)
        if type == 0:
            # DetAggRule
            # Step 1: get filter cond
            cond = sample_cond(cond_depth, num_features)

            # Step 2: get map function
            map_fun = sample_map_fun(num_features)

            rules.append( DetAggRule(cond, map_fun) )
        else:
            # NonDetRule
            cond = sample_cond(cond_depth, num_features)
            rules.append( NonDetRule(cond) )

    return ProgramPolicy(rules)


def sample_cond(depth = 2, num_features = 12):
    ctype = random.randint(0, 2)
    if depth == 1:
        ctype = 0
    if ctype == 0:
        # linear cond
        return sample_linear_cond(num_features)
    elif ctype == 1:
        # and cond
        cond1 = sample_cond(depth - 1, num_features)
        cond2 = sample_cond(depth - 1, num_features)
        return AndCond(cond1, cond2)
    else:
        # or cond
        cond1 = sample_cond(depth - 1, num_features)
        cond2 = sample_cond(depth - 1, num_features)
        return OrCond(cond1, cond2)

def sample_linear_cond(num_features):
    weights = torch.zeros(num_features + 1, dtype = torch.float32).to(dev)

    idx = random.randint(0, num_features - 1)
    sign = random.choice([-1, 1])
    const = random.uniform(-1, 1)
    weights[idx] = sign
    weights[-1] = const

    return LinearCond(weights)

def sample_map_fun(num_features):
    weights = torch.zeros(num_features + 1, dtype = torch.float32).to(dev)

    idx = random.randint(0, num_features - 1)
    sign = random.choice([-1, 1])
    weights[idx] = sign

    return LinearMap(weights)


def test_prog_policy():
    cond1 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).to(dev), ignore_self = True)
    lmap1 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r1 = DetAggRule(cond1, lmap1)

    cond2 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32).to(dev), ignore_self = True)
    lmap2 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r2 = DetAggRule(cond2, lmap2)

    cond3 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -3/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond4 = LinearCond(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond5 = AndCond(cond3, cond4)
    r3 = NonDetRule(cond5)


    return ProgramPolicy([r1, r3])

def test_prog_policy1():
    cond1 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).to(dev), ignore_self = True)
    lmap1 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r1 = DetAggRule(cond1, lmap1)

    cond2 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32).to(dev), ignore_self = True)
    lmap2 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r2 = DetAggRule(cond2, lmap2)

    cond3 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -3/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond4 = LinearCond(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond5 = AndCond(cond3, cond4)
    r3 = NonDetRule(cond5)


    return ProgramPolicy([r1, r3])


def test_prog_policy_edgeonly():
    cond1 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).to(dev), ignore_self = True)
    lmap1 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r1 = DetAggRule(cond1, lmap1)

    cond3 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, -3/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond4 = LinearCond(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, -7/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond5 = AndCond(cond3, cond4)
    r3 = NonDetRule(cond5)


    return ProgramPolicy([r1, r3])

def test_prog_policy3():
    cond1 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).to(dev), ignore_self = True)
    lmap1 = LinearMap(torch.tensor([0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=torch.float32).to(dev))
    r1 = DetAggRule(cond1, lmap1)



    cond3 = LinearCond(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, -2/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond4 = LinearCond(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, -7/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond5 = AndCond(cond3, cond4)

    cond6 = LinearCond(torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 2/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond7 = AndCond(cond4, cond6)

    cond8 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, -2/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond9 = AndCond(cond4, cond8)

    cond10 = LinearCond(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 2/20.0], dtype=torch.float32).to(dev), ignore_self = True)
    cond11 = AndCond(cond4, cond10)


    r2 = NonDetRule(cond5)
    r3 = NonDetRule(cond7)
    r4 = NonDetRule(cond9)
    r5 = NonDetRule(cond11)


    return ProgramPolicy([r1, r2, r3, r4, r5])

def rand_prog_policy(num_rules):
    rules = []
    for i in range(num_rules):
        cond = LinearCond(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).to(dev), ignore_self = True)
        r = NonDetRule(cond)
        rules.append(r)
    return ProgramPolicy(rules)
