import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import gca
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import pickle
import json

from envs.formation_flying import *
from main.policy import neural_policy_bptt
from main.policy import neural_policy_bptt_batch

import main.utils as utils

from tasks.task import *

from sklearn import tree
from sklearn.utils import shuffle
from joblib import dump, load

from main.policy import program_attn_policy
from main.policy import prob_program_attn_policy

from main.policy.program_attn_policy import *
from main.policy.prob_program_attn_policy import *



def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--result_dir', type=str, default='test', help='folder to store results')

    parser.add_argument('--task', type=str, default="random_cross")
    parser.add_argument('--num_groups', type=int, default=3)
    parser.add_argument('--alg', type = str, default="mcmc", help = "Program synthesis algorithm, can be mcmc or enum")


    parser.add_argument('--policy', type=str, default='transformer',
        help='type of policy, can be transformer')
    parser.add_argument('--num_agents', type=int, default=20,
        help='Number of agents per group')
    parser.add_argument('--collision_penalty', type=float, default=0.1,
        help='Collision penalty (per collision)')
    parser.add_argument('--collision_dist', type=float, default=0.1,
        help='Collision distance')
    parser.add_argument('--transformer_model_dim', type=int, default=16, metavar='N',
        help='Transformer hidden dimension')
    parser.add_argument('--max_step', type=int, default=200,
        help='Max step')

    parser.add_argument('--deg_loss_wt', type=float, default=1.0,
        help='Weight of the degree loss')
    parser.add_argument('--mcmc_steps', type=int, default=2000,
        help='Number of MCMC steps')
    parser.add_argument('--mcmc_restart', type=int, default=500,
        help='Number of MCMC steps to restart')
    parser.add_argument('--final_steps', type=int, default=1000,
        help='Number of final MCMC steps')
    parser.add_argument('--max_deg', type=int, default=2,
        help='Max comms degree')
    parser.add_argument('--num_rules', type=int, default=2,
        help='Max comms degree')

    parser.add_argument('--env', type=str, default='cross_arrow',
        help='type of env, can be cross_arrow, two_groups')
    parser.add_argument('--sep_self_edge', action='store_true', default=False,
        help='Separate self edge embedding')
    parser.add_argument('--gap', type=float, default=3.0, help='Average gap for initial location, for two_groups crossing. Times the collision_dist.')
    parser.add_argument('--col_loss', type=str, default='ramp',
        help='type of collision loss, can be ramp, exp')
    parser.add_argument('--reward', type=str, default='total',
        help='reward structure. Can be total/change')
    parser.add_argument('--clip_speed', action='store_true', default=False,
        help='Clipping actual speed')
    parser.add_argument('--load_all_args', action='store_true', default=False)

    parser.add_argument('--use_suffix', action='store_true', default=False)
    parser.add_argument('--use_internal_state', action='store_true', default=False,
        help='Use internal state in transformer')
    parser.add_argument('--use_sigmoid', action = "store_true", default = False, help = "Sigmoid for action")

    parser.add_argument('--use_soft_with_prog', action='store_true', default=False, help='Use soft attention together with prog_attn')

    parser.add_argument('--noise_scale', type=float, default=0.0,
        help='Noise scale for observations')

    parser.add_argument('--suffix', type=str, default='')

    parser.add_argument('--grid_l', type=float, default=14.0,
        help='Grid cell size')

    parser.add_argument('--settings', type=str, default='orig',
        help='Settings used to train. Can be orig, high, balance, single')
    parser.add_argument('--sep_hops', action = "store_true", default = False, help='Learn prog for each hop separately')
    parser.add_argument('--init_gap', type = int, default = 0)
    parser.add_argument('--loss_type', type = str, default = "both", help = "can be both, action, dist")

    parser.add_argument('--feature', type=str, default='orig',
        help='Program features. Can be orig, quadratic, goal, both')
    parser.add_argument('--no_random', action = "store_true", default = False, help = "Learn programs without random")

    return parser.parse_args()


def update_args_using_saved(args, saved_args):
    args.policy = saved_args.policy
    args.collision_dist = saved_args.collision_dist
    args.collision_penalty = saved_args.collision_penalty
    args.transformer_model_dim = saved_args.transformer_model_dim
    #args.gap = saved_args.gap
    args.col_loss = saved_args.col_loss
    #args.reward = saved_args.reward
    args.clip_speed = saved_args.clip_speed

    args.sep_key_val = saved_args.sep_key_val
    args.hops = saved_args.hops
    args.dropout_model = saved_args.dropout_model
    args.dropout_percent = saved_args.dropout_percent

    if args.alg == "pg":
        args.with_sigma = saved_args.with_sigma

    if hasattr(saved_args, 'use_sigmoid'):
        args.use_sigmoid = saved_args.use_sigmoid

    if hasattr(saved_args, 'self_attn'):
        args.self_attn = saved_args.self_attn
        args.dist_based_hard = saved_args.dist_based_hard
        args.comm_deg = saved_args.comm_deg
        args.only_obs_edge = saved_args.only_obs_edge

    if hasattr(saved_args, 'sep_self_edge'):
        args.sep_self_edge = saved_args.sep_self_edge

    if hasattr(saved_args, 'noise_scale'):
        args.noise_scale = saved_args.noise_scale

    if hasattr(saved_args, 'use_internal_state'):
        args.use_internal_state = saved_args.use_internal_state
    if hasattr(saved_args, 'grid_l'):
        args.grid_l = saved_args.grid_l
    if hasattr(saved_args, 'init_gap'):
        args.init_gap = saved_args.init_gap
    if hasattr(saved_args, 'loss_type'):
        args.loss_type = saved_args.loss_type

    if args.load_all_args:
        args.num_agents = saved_args.num_agents
        args.env = saved_args.env
        args.hard_attn = saved_args.hard_attn

    return args


def main():
    # Set up parameters
    args = get_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.result_dir)

    device = torch.device("cuda")

    saved_args_path = os.path.join(results_dir, 'args.pickle')
    if os.path.exists(saved_args_path):
        with open(saved_args_path, 'rb') as arg_f:
            saved_args = pickle.load(arg_f)
        args = update_args_using_saved(args, saved_args)

    # Set up task and env
    if args.task == 'random_cross':
        task_args = {
            'box_width_ratio': 8.0,
            'l': 7.0,
            'num_groups': args.num_groups,
        }
    elif args.task == 'random_grid':
        task_args = {
            'box_width_ratio': 8.0,
            'l': args.grid_l,
        }
    elif args.task == "unlabeled":
        task_args = {}
    elif args.task == "traffic":
        task_args = {}
    else:
        print("Task name not supported")
        assert(False)

    task = get_task_new(args.task, task_args)

    env_args = {
        'task': task,
        'device': device,
        'num_agents': args.num_agents,
        'collision_penalty': args.collision_penalty,
        'collision_dist': args.collision_dist,
        'comm_dropout_model': args.dropout_model,
        'comm_dropout_percent': args.dropout_percent,
        'col_loss': args.col_loss,
        'clip_speed': args.clip_speed,
        'init_gap' : args.init_gap,
        'loss_type' : args.loss_type,
        'noise_scale': args.noise_scale,
        'prog_feature': args.feature,
    }

    if not args.self_attn:
        env_args['comm_dropout_model'] = 'no_self'

    exp_env = gym.make(args.env, args=env_args)
    exp_env._max_episode_steps = args.max_step

    input_dim = exp_env.n_features
    edge_dim = 2
    nhead = 1

    # Load policy
    filename = os.path.join(results_dir, "model.pt")
    if args.policy == 'transformer_edge':
        policy_settings = {
            'transformer_model_dim' : args.transformer_model_dim,
            'nhead' : nhead,
            'transformer_fc_dim' : args.transformer_model_dim,
            'device' : device,
            'input_dim' : input_dim,
            'edge_dim' : edge_dim,
            'sep_self_edge' : args.sep_self_edge,
            'n_hops' : args.hops,
            'with_sigma' : args.alg == "pg",
            'sep_key_val' : args.sep_key_val,
            'hard_attn' : False,
            'hard_attn_deg' : 2,
            'rand_hard_attn' : False,
            'train_attention' : True,
            'only_obs_edge' : args.only_obs_edge,
            'num_out' : exp_env.nu,
            'use_internal_state': args.use_internal_state,
            'use_sigmoid' : args.use_sigmoid,

        }
        # initialize policy
        policy = neural_policy_bptt.TransformerEdgePolicy(policy_settings).to(device)
        policy.load_state_dict(torch.load(filename))

        policy1 = neural_policy_bptt_batch.TransformerEdgePolicy(policy_settings).to(device)
        policy1.load_state_dict(torch.load(filename))

    else:
        print("Policy not supported")
        return

    policy.eval()
    policy1.eval()


    # Get data set
    if args.settings == 'orig':
        test_settings = get_trainprog_settings(args.task, args.num_groups)
    elif args.settings == 'high':
        test_settings = get_trainprog_settings_high(args.task, args.num_groups)
    elif args.settings == 'balance':
        test_settings = get_trainprog_settings_balance(args.task, args.num_groups)
    elif args.settings == 'single':
        test_settings = get_trainprog_settings_single(args.task, args.num_groups)
    else:
        raise RuntimeError("Train prog settings not supported")

    num_batches = max(10 // len(test_settings), 1)
    #num_batches = 3 # per setting
    batch_size  = 30 # per group per setting
    data_list = []
    for setting in test_settings:
        reset_args = {
                'num_agents': args.num_agents,
                'setting': setting,
        }
        for g in range(num_batches):
            print("Setting:", setting, "batch:", g)
            data = collect_data(exp_env, policy, reset_args, num_episodes = batch_size)
            data_list.append(data)

    #args.self_attn = True
    if args.self_attn:
        # learn a dt for self edges
        print("Training decision tree for self edges")
        dt = train_dt_self_edge(data_list, results_dir)

        #dt_file = os.path.join(results_dir, 'dt.joblib')
        #dt = load(dt_file)

        # Learn program policy for non self edge attention
        print("Training program policy")
        if args.alg == "mcmc":
            do_mcmc(data_list, exp_env, policy1, results_dir, dt=dt, only_obs_edge=args.only_obs_edge, steps=args.mcmc_steps, num_steps_per_init=args.mcmc_restart, final_steps=args.final_steps, deg_loss_wt=args.deg_loss_wt, max_deg=args.max_deg, num_rules=args.num_rules, suffix=args.suffix, soft_with_prog=args.use_soft_with_prog, num_progs = args.hops)

        elif args.alg == "enum":
            do_enum(data_list, exp_env, policy1, results_dir, dt=dt, only_obs_edge=args.only_obs_edge, steps=args.mcmc_steps, deg_loss_wt=args.deg_loss_wt, max_deg=args.max_deg, num_rules=args.num_rules, suffix=args.suffix, soft_with_prog=args.use_soft_with_prog, num_progs = args.hops)
        else:
            print("Unknown alg")
            assert(False)

    else:
        # Learn program policy for non self edge attention
        print("Training program policy")
        if args.alg == "mcmc":
            if not args.sep_hops or args.hops == 1:
                do_mcmc(data_list, exp_env, policy1, results_dir, only_obs_edge=args.only_obs_edge, steps=args.mcmc_steps, num_steps_per_init=args.mcmc_restart, final_steps=args.final_steps,  deg_loss_wt=args.deg_loss_wt, max_deg=args.max_deg, num_rules=args.num_rules, suffix=args.suffix, soft_with_prog=args.use_soft_with_prog, num_progs = args.hops, with_non_det = not args.no_random)
            else:
                for hop in range(args.hops):
                    print("Learing for hop %i"%hop)
                    do_mcmc(data_list, exp_env, policy1, results_dir, only_obs_edge=args.only_obs_edge, steps=args.mcmc_steps, num_steps_per_init=args.mcmc_restart, final_steps=args.final_steps,  deg_loss_wt=args.deg_loss_wt, max_deg=args.max_deg, num_rules=args.num_rules, suffix=args.suffix + "_hop%i"%hop, soft_with_prog=args.use_soft_with_prog, num_progs = 1, hop = hop, with_non_det = not args.no_random)



        elif args.alg == "enum":
            #assert False, "Not supported no self_attn for enum"
            do_enum(data_list, exp_env, policy1, results_dir, only_obs_edge=args.only_obs_edge, steps=args.mcmc_steps,  deg_loss_wt=args.deg_loss_wt, max_deg=args.max_deg, num_rules=args.num_rules, suffix=args.suffix, soft_with_prog=args.use_soft_with_prog, num_progs = args.hops)
        else:
            print("Unknown alg")
            assert(False)


def train_dt_self_edge(data_list, results_dir):
    train_X = []
    train_Y = []
    for data in data_list:
        inputs, edge_inputs, outputs, _ = data
        S, N, d = inputs.shape
        train_X.extend(inputs.reshape((S*N, d)).data.cpu().numpy())
        Y, X = np.meshgrid(np.arange(N), np.arange(S))
        X = X.reshape(N*S)
        Y = Y.reshape(N*S)
        max_weights = torch.max(outputs, dim=-1,keepdim=True).values
        outputs =(outputs/max_weights)/2.0
        diag = outputs[X, Y, Y]
        diag[diag > 0.4] = 1
        diag[diag <= 0.4] = 0
        train_Y.extend(diag.data.cpu().numpy())

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    neg_idx = np.nonzero(train_Y == 0)
    pos_idx = np.nonzero(train_Y)
    pos_X = train_X[pos_idx]
    neg_X = train_X[neg_idx]

    l_pos_X = []
    l_neg_X = []

    max_len = max(len(pos_X), len(neg_X))

    while len(l_neg_X) < len(pos_X):
        l_neg_X.extend(neg_X)
    l_neg_X = l_neg_X[0:max_len]
    while len(l_pos_X) < len(neg_X):
        l_pos_X.extend(pos_X)
    l_pos_X = l_pos_X[0:max_len]

    assert(len(l_pos_X) == len(l_neg_X))

    train_X = []
    train_Y = []
    train_X.extend(l_pos_X)
    train_Y.extend([1 for _ in l_pos_X])
    train_X.extend(l_neg_X)
    train_Y.extend([0 for _ in l_neg_X])

    train_X, train_Y = shuffle(train_X, train_Y, random_state=0)
    clf = tree.DecisionTreeClassifier(max_depth = 3)
    clf = clf.fit(train_X, train_Y)

    filename = os.path.join(results_dir, 'dt.joblib')
    dump(clf, filename)

    print("Verifying")
    predicted = clf.predict(train_X)
    correct = np.sum(predicted == train_Y)
    total = len(predicted)

    train_Y = np.array(train_Y)


    pos_correct = np.sum(predicted[train_Y == 1] == train_Y[train_Y == 1])
    pos_total = len(train_Y[train_Y == 1])


    print("Train accuracy of the network: %d/%d %d %%"%(correct, total, 100.0*correct/total))
    print("Train pos accuracy of the network: %d/%d %d %%"%(pos_correct, pos_total, 100.0*pos_correct/pos_total))


    return clf

def acceptance(metric, metric_new):
    if metric_new < metric:
        return True
    else:
        a = np.random.uniform(0, 1)
        diff = metric - metric_new
        return a < torch.exp(10.0*diff/abs(metric))



def do_mcmc(data_list, env, policy, results_dir, dt=None, only_obs_edge=False, steps=2000, num_steps_per_init=500, final_steps=1000, deg_loss_wt=1.0, max_deg=2, num_progs = 1, num_rules=2, suffix='', soft_with_prog=False, hop = None, with_non_det = True):

    log_file_name = 'prog_log_{}.txt'.format(suffix)
    outputManager = utils.OutputManager(results_dir, log_file_name)

    min_metric = torch.tensor([1e20,1e20,1e20], dtype = torch.float32).to(dev)
    min_progs = None
    min_prob_progs = None


    num_features = env.n_prog_features

    #num_steps_per_init = 1
    num_init = steps//num_steps_per_init

    for k in range(num_init):
        prob_progs = []
        progs = []
        for j in range(num_progs):
            prob_prog = ProbProgramPolicy.random(num_rules = num_rules, cond_depth = 2, num_features = num_features, with_non_det = with_non_det)
            prog = prob_prog.sample_prog()
            prob_progs.append(prob_prog)
            progs.append(prog)

        for prog in progs:
            if not dt:
                # No self_attn
                prog.set_no_self()
            else:
                prog.set_dt_self_edge(dt)

            if only_obs_edge:
                prog.set_only_edge()

        metric = get_data_metric(progs, data_list, env,policy, deg_loss_wt=deg_loss_wt, max_deg=max_deg, soft_with_prog=soft_with_prog, hop = hop )

        if metric.sum() < min_metric.sum():
            min_metric = metric
            min_progs = progs
            min_prob_progs = prob_progs

        for i in tqdm(range(num_steps_per_init)):
            prob_progs_new = []
            progs_new = []
            for j in range(num_progs):
                prob_prog_new = prob_progs[j].sample_next(std = 0.2)
                prog_new = prob_prog_new.sample_prog()
                prob_progs_new.append(prob_prog_new)
                progs_new.append(prog_new)
            for prog_new in progs_new:
                if not dt:
                    # No self_attn
                    prog_new.set_no_self()
                else:
                    prog_new.set_dt_self_edge(dt)

                if only_obs_edge:
                    prog_new.set_only_edge()

            metric_new = get_data_metric(progs_new, data_list, env, policy, deg_loss_wt=deg_loss_wt, max_deg=max_deg, soft_with_prog=soft_with_prog, hop = hop)

            if metric_new.sum() < min_metric.sum():
                min_metric = metric_new
                min_progs = progs_new
                min_prob_progs = prob_progs_new

            if acceptance(metric.sum(), metric_new.sum()):
                prob_progs = prob_progs_new
                metric = metric_new

            if i % 100 == 0 or i == num_steps_per_init - 1:
                outputManager.say("Step {} ".format(i))
                outputManager.say(min_metric)
                outputManager.say(min_metric.sum())
                outfile = open(os.path.join(results_dir, 'prog.txt'+suffix), 'w')
                for min_prog in min_progs:
                    outfile.write(repr(min_prog))
                    outfile.write("\n")
                outfile.close()

                with open(os.path.join(results_dir, 'prob_prog'+suffix+'.pickle'), 'wb') as outf:
                    pickle.dump(min_prob_progs, outf)

    # Run final steps from best prob_progs seen so far
    prob_progs = min_prob_progs
    progs = min_progs
    metric = get_data_metric(progs, data_list, env,policy, deg_loss_wt=deg_loss_wt, max_deg=max_deg, soft_with_prog=soft_with_prog)

    for i in tqdm(range(final_steps)):
        prob_progs_new = []
        progs_new = []
        for j in range(num_progs):
            prob_prog_new = prob_progs[j].sample_next(std = 0.2)
            prog_new = prob_prog_new.sample_prog()
            prob_progs_new.append(prob_prog_new)
            progs_new.append(prog_new)
        for prog_new in progs_new:
            if not dt:
                # No self_attn
                prog_new.set_no_self()
            else:
                prog_new.set_dt_self_edge(dt)

            if only_obs_edge:
                prog_new.set_only_edge()

        metric_new = get_data_metric(progs_new, data_list, env, policy, deg_loss_wt=deg_loss_wt, max_deg=max_deg, soft_with_prog=soft_with_prog)

        if metric_new.sum() < min_metric.sum():
            min_metric = metric_new
            min_progs = progs_new
            min_prob_progs = prob_progs_new

        if acceptance(metric.sum(), metric_new.sum()):
            prob_progs = prob_progs_new
            metric = metric_new

        if i % 100 == 0 or i == final_steps - 1:
            outputManager.say("Step {} ".format(i))
            outputManager.say(min_metric)
            outputManager.say(min_metric.sum())
            outfile = open(os.path.join(results_dir, 'prog.txt'+suffix), 'w')
            for min_prog in min_progs:
                outfile.write(repr(min_prog))
                outfile.write("\n")
            outfile.close()

            with open(os.path.join(results_dir, 'prob_prog'+suffix+'.pickle'), 'wb') as outf:
                pickle.dump(min_prob_progs, outf)


def do_enum(data_list, env, policy, results_dir, dt=None, only_obs_edge=False, steps=2000, deg_loss_wt=1.0, max_deg=2, num_progs = 1, num_rules=2, suffix='', soft_with_prog=False):
    min_metric = torch.tensor([1e20,1e20,1e20], dtype = torch.float32).to(dev)
    min_progs = None
    num_features = env.n_prog_features


    for i in range(5000):
        progs = []
        for j in range(num_progs):
            prog = sample_prog_policy(num_rules = num_rules, cond_depth = 2, num_features = num_features)
            if not dt:
                # No self_attn
                prog.set_no_self()
            else:
                prog.set_dt_self_edge(dt)

            if only_obs_edge:
                prog.set_only_edge()
            progs.append(prog)

        metric = get_data_metric(progs, data_list, env, policy, deg_loss_wt=deg_loss_wt, max_deg=max_deg, soft_with_prog=soft_with_prog)

        if metric.sum() < min_metric.sum():
            min_metric = metric
            min_progs = progs

        if i % 100 == 0:
            print(min_metric)
            outfile = open(os.path.join(results_dir, 'prog.txt'), 'w')
            for min_prog in min_progs:
                    outfile.write(repr(min_prog))
                    outfile.write("\n")
            outfile.close()

def get_data_metric(progs, data_list, env, policy, deg_loss_wt=1.0, max_deg=2, soft_with_prog=False, hop = None):
    if hop == None:
        policy.use_prog_for_attn(progs, use_soft_with_prog=soft_with_prog)
    else:
        policy.use_prog_for_attn1(progs, hop, use_soft_with_prog=soft_with_prog)


    attn_cost = 0
    action_cost = 0
    total_rank_cost = 0
    for data in data_list:
        inputs, edge_inputs, prog_inputs, internal_states, out_attn, out_actions = data
        N = len(inputs[0])
        comm_weights = torch.ones((N, N)).to(dev)*100

        if policy.use_internal_state:
            policy.internal_state = internal_states

        actions, attns = policy(inputs, edge_inputs, prog_input = prog_inputs, need_weights = True)
        actions = env.preprocess_action(actions)


        #sub_weights = out_attn[attn > 1e-2]
        #attn_cost += - sub_weights.sum()

        action_cost += torch.norm(actions - out_actions, p=1, dim=-1).sum()

        for attn in attns:
            S, N, _ = attn.shape

            Y, X = np.meshgrid(np.arange(N), np.arange(S))
            X = X.reshape(N*S)
            Y = Y.reshape(N*S)

            attn[X, Y, Y] = 0
            attn[attn > 1e-2] = 1

            in_cost = (attn.sum(axis = 2).max(axis = -1).values - max_deg).clamp(0)
            out_cost = (attn.sum(axis = 1).max(axis = -1).values - max_deg).clamp(0)
            rank_cost = (in_cost + out_cost).sum()

            total_rank_cost += rank_cost

    return torch.tensor([attn_cost*0.0, action_cost, total_rank_cost*deg_loss_wt*1.0], dtype = torch.float32).to(dev)


def collect_data(env, policy, reset_args, num_episodes = 100):
    inputs = []
    edge_inputs = []
    prog_inputs = []
    internal_states = []
    outputs = []
    actions = []

    for episode in range(num_episodes):
        env.set_reset_args(reset_args)
        state = env.reset() # Reset environment and record the starting state
        max_step = env._max_episode_steps
        policy.reset_internal_state()
        for j in range(max_step):
            obs = env.get_obs(state)
            if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy):
                relative_pos = env.get_relative_pos(state)
                comm_weights = env.get_comm_weights(state)
                prog_input = env.get_prog_features(state, relative_pos)
                if policy.use_internal_state:
                    internal_state = policy.internal_state
                    if internal_state == None:
                        internal_state = torch.zeros(len(state), policy.model_dim, dtype = torch.float32).to(env.device)
                    internal_states.append(internal_state.unsqueeze(0))
                action, att_weights = policy(obs, relative_pos, comm_weights, need_weights = True)
                action_c = env.preprocess_action(action)

                att_weights = att_weights[0][0].detach()

                inputs.append(obs.detach().unsqueeze(0))
                edge_inputs.append(relative_pos.detach().unsqueeze(0))
                prog_inputs.append(prog_input.detach())
                outputs.append(att_weights.detach().unsqueeze(0))
                actions.append(action_c.detach().unsqueeze(0))

            else:
                assert(False)

            state = env.step1(state, action)
            if env.done(state):
                break

    return torch.cat(inputs, dim = 0), torch.cat(edge_inputs, dim = 0), torch.cat(prog_inputs, dim = 0), torch.cat(internal_states, dim = 0) if len(internal_states) > 0 else None, torch.cat(outputs, dim = 0), torch.cat(actions, dim = 0)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
