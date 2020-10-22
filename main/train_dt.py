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

import main.utils

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

    parser.add_argument('--max_in_deg', type=int, default=2,
        help='Max comms degree')
    parser.add_argument('--max_out_deg', type=int, default=2,
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
    if hasattr(saved_args, 'feature'):
        args.feature = saved_args.feature 

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


    else:
        print("Policy not supported")
        return

    policy.eval()


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


    for hop in range(args.hops):
        print("Hop", hop)
        # get train data 
        print("Collecting train data")
        num_batches = max(10 // len(test_settings), 1)
        #num_batches = 3 # per setting
        batch_size  = 30 # per group per setting
        data_list = []
        for setting in test_settings:
            reset_args = {
                    'num_agents': args.num_agents,
                    'setting': setting,
                    #'num_groups' : gr,
            }
            for g in range(num_batches):
                print("Setting:", setting, "batch:", g)
                data = collect_data(exp_env, policy, reset_args, hop = hop, num_episodes = batch_size)
                data_list.append(data)

        train_X, train_Y = preprocess_data(data_list, args.max_in_deg, args.max_out_deg, device)

        # get test data 
        print("Collecting validation data")
        num_batches = max(10 // len(test_settings), 1)
        #num_batches = 3 # per setting
        batch_size  = 10 # per group per setting
        test_data_list = []
        for setting in test_settings:
            reset_args = {
                    'num_agents': args.num_agents,
                    'setting': setting,
                    #'num_groups' : gr,
            }
            for g in range(num_batches):
                print("Setting:", setting, "batch:", g)
                data = collect_data(exp_env, policy, reset_args, num_episodes = batch_size)
                test_data_list.append(data)

        test_X, test_Y = preprocess_data(test_data_list, args.max_in_deg, args.max_out_deg, device)

        print("Training decision tree")
        dt = train_dt(train_X, train_Y, test_X, test_Y, results_dir, hop)


def preprocess_data(data_list, max_in_deg, max_out_deg, device):
    train_X = []
    train_Y = []
    for data in data_list:
        _, _, prog_inputs, _, out_attn, _ = data
        S, N, _, d = prog_inputs.shape
        train_X.extend(prog_inputs.reshape((S*N*N, d)).data.cpu().numpy())

        in_degree_so_far = torch.zeros(S, N, dtype = torch.float32).to(device)
        out_degree_so_far = torch.zeros(S, N, dtype = torch.float32).to(device)


        undir_edge_weights = out_attn.reshape(S, N*N)
        sort_idx = undir_edge_weights.argsort(descending=True)


        tensor = torch.ones((2,), dtype=torch.float32)
        new_attn_weights = tensor.new_full((S, N, N), float(0),  dtype=torch.float32)
        for s in range(S):
            for idx in sort_idx[s]:
                x = idx//N
                y = idx%N
                if  in_degree_so_far[s][x] < max_in_deg and out_degree_so_far[s][y] < max_out_deg:
                    new_attn_weights[s][x][y] = 1

                    in_degree_so_far[s][x] += 1
                    out_degree_so_far[s][y] += 1 


        train_Y.extend(new_attn_weights.view(S*N*N).data.cpu().numpy())

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    neg_idx = np.nonzero(train_Y == 0)
    pos_idx = np.nonzero(train_Y)
    pos_X = train_X[pos_idx]
    neg_X = train_X[neg_idx]
    print(len(pos_X), len(neg_X))

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

    return train_X, train_Y


def train_dt(train_X, train_Y, test_X, test_Y, results_dir, hop):
    max_clf = None 
    max_correct = -1
    for depth in [3, 5, 10, 20]:
        print("Training dt for depth %i"%depth)
        clf = tree.DecisionTreeClassifier(max_depth = depth)
        clf = clf.fit(train_X, train_Y)

        filename = os.path.join(results_dir, 'dt_%ihop_%i.joblib'%(hop, depth))
        dump(clf, filename)

        print("Validating")
        predicted = clf.predict(train_X)
        correct = np.sum(predicted == train_Y)
        total = len(predicted)

        train_Y = np.array(train_Y)


        pos_correct = np.sum(predicted[train_Y == 1] == train_Y[train_Y == 1])
        pos_total = len(train_Y[train_Y == 1])


        print("Train accuracy of the network: %d/%d %d %%"%(correct, total, 100.0*correct/total))
        print("Train pos accuracy of the network: %d/%d %d %%"%(pos_correct, pos_total, 100.0*pos_correct/pos_total))

        predicted = clf.predict(test_X)
        correct = np.sum(predicted == test_Y)
        total = len(predicted)

        test_Y = np.array(test_Y)


        pos_correct = np.sum(predicted[test_Y == 1] == test_Y[test_Y == 1])
        pos_total = len(test_Y[test_Y == 1])


        print("Test accuracy of the network: %d/%d %d %%"%(correct, total, 100.0*correct/total))
        print("Test pos accuracy of the network: %d/%d %d %%"%(pos_correct, pos_total, 100.0*pos_correct/pos_total))

        if correct > max_correct:
            max_correct = correct 
            max_clf = clf 

    filename = os.path.join(results_dir, 'dt_%ihop.joblib'%hop)
    dump(max_clf, filename)
    return max_clf


def collect_data(env, policy, reset_args, hop = 0, num_episodes = 100):
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

                att_weights = att_weights[hop][0].detach()

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
