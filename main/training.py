import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import random
import pickle

from main.policy import neural_policy_bptt
import main.utils as putils

from envs.formation_flying import *
from tasks.task import *

from joblib import dump, load

from main.policy import program_attn_policy
from main.policy.program_attn_policy import *


def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='enable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--result_dir', type=str, default='test', help='folder to store results')
    parser.add_argument('--episode', type=int, default=15000, metavar='N',
        help='number of episode to train (default: 15000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
        help='learning rate (default: 1e-3)')
    parser.add_argument('--policy', type=str, default='transformer',
        help='type of policy, can be transformer')
    parser.add_argument('--env', type=str, default="FormationTorch-v3")
    parser.add_argument('--num_agents', type=int, default=20,
        help='Number of agents')
    parser.add_argument('--max_step', type=int, default=200,
        help='Max step')

    parser.add_argument('--collision_penalty', type=float, default=0.1,
        help='Collision penalty (per collision)')
    parser.add_argument('--collision_dist', type=float, default=0.1,
        help='Collision distance')
    parser.add_argument('--transformer_model_dim', type=int, default=16, metavar='N',
        help='Transformer hidden dimension')
    parser.add_argument('--clip_speed', action='store_true', default=False,
        help='Clipping actual speed')
    parser.add_argument('--col_loss', type=str, default='ramp',
        help='type of collision loss, can be ramp, exp')
    parser.add_argument('--alg', type = str, default="bptt", help = "Reinforcement learning algorithm, can be bptt or pg")
    parser.add_argument('--reward', type=str, default='total',
        help='reward structure. Can be total/change')

    parser.add_argument('--task', type=str, default='cross',
        help='type of env')
    parser.add_argument('--dropout_model', type = str, default=None, help = "model for communication dropouts, can be random")
    parser.add_argument('--dropout_percent', type = int, default=50, help = "Percent of communication dropouts, 0 to 100")
    parser.add_argument('--hops', type = int, default=1, help = "Number of hops of communications per time step, can be 1 or 2")
    parser.add_argument('--sep_key_val', action = 'store_true', default=False, help="Separate encodings for key and value in transformer")
    parser.add_argument('--use_internal_state', action = 'store_true', default=False, help="Separate encodings for key and value in transformer")

    parser.add_argument('--retrain_with_prog_attn', action = "store_true", default=False)
    parser.add_argument('--retrain_with_dt_attn', action = "store_true", default=False)
    parser.add_argument('--num_groups', type = int, default=4)

    parser.add_argument('--self_attn', action = 'store_true', default=False, help='To have self attention in transformer')
    parser.add_argument('--dist_based_hard', action = 'store_true', default=False, help='To use distance based degree constraint for transformer')
    parser.add_argument('--comm_deg', type=int, default=2, help='Distance based degree constraint for transformer')
    parser.add_argument('--only_obs_edge', action='store_true', default=False, help='To have transformer only use edge feature to decide who to attend to')
    parser.add_argument('--use_sigmoid', action = "store_true", default = False, help = "Sigmoid for action")
    parser.add_argument('--detach_state', action = "store_true", default = False, help = "Detach state after every step of the simulation")
    parser.add_argument('--use_soft_with_prog', action='store_true', default=False, help='Use soft attention together with prog_attn')

    parser.add_argument('--suffix', type=str, default='',
        help='Suffix for saved program_policy')

    parser.add_argument('--grid_l', type=float, default=14.0,
        help='Grid cell size')
    parser.add_argument('--ordering_scheme', type = str , default = "min", help = "Ordering scheme for unassigned task, can be fixed or min")
    parser.add_argument('--sep_hops', action = "store_true", default = False, help='Learn prog for each hop separately')

    parser.add_argument('--noise_scale', type=float, default=0.0,
        help='Noise scale for observations')

    # traffic junction task params
    #parser.add_argument('--add_prob', type = float, default = 0.2)
    #parser.add_argument('--re_add_agents', action = "store_true", default = False)
    parser.add_argument('--loss_type', type = str, default = "action", help = "can be both, action, dist")
    parser.add_argument('--init_gap', type = int, default = 0)

    parser.add_argument('--feature', type=str, default='orig',
        help='Program features. Can be orig, quadratic, goal, both')

    return parser.parse_args()


def plot_stats_over_episodes(data, filename):
    fig = plt.figure()
    x = np.linspace(0,len(data),len(data))
    plt.plot(x,data)
    plt.savefig(filename)
    plt.close()




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

    if hasattr(saved_args, 'use_internal_state'):
        args.use_internal_state = saved_args.use_internal_state

    if hasattr(saved_args, 'init_gap'):
        args.init_gap = saved_args.init_gap

    if hasattr(saved_args, 'loss_type'):
        args.loss_type = saved_args.loss_type


    if hasattr(saved_args, 'noise_scale'):
        args.noise_scale = saved_args.noise_scale

    if hasattr(saved_args, 'detach_state'):
        args.detach_state = saved_args.detach_state

    return args


def main():
    # Set up parameters
    args = get_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.result_dir)
    if not args.retrain_with_prog_attn and not args.retrain_with_dt_attn:
        putils.create_dir(results_dir)

    suffix = args.suffix
    if args.retrain_with_prog_attn:
        suffix += "_retrained"
        saved_args_path = os.path.join(results_dir, 'args.pickle')
        if os.path.exists(saved_args_path):
            with open(saved_args_path, 'rb') as arg_f:
                saved_args = pickle.load(arg_f)
            args = update_args_using_saved(args, saved_args)
    if args.retrain_with_dt_attn:
        suffix += "_dt_retrained"
        saved_args_path = os.path.join(results_dir, 'args.pickle')
        if os.path.exists(saved_args_path):
            with open(saved_args_path, 'rb') as arg_f:
                saved_args = pickle.load(arg_f)
            args = update_args_using_saved(args, saved_args)


    outputManager = putils.OutputManager(results_dir)
    outputManager.say("Args:")
    outputManager.say(args)

    with open(os.path.join(results_dir, 'args%s.pickle'%suffix), 'wb') as arg_f:
        pickle.dump(args, arg_f)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    elif args.task == 'unlabeled':
        task_args = {}
    elif args.task == "traffic":
        task_args = {}
    elif args.task == 'random_group':
        task_args = {
            'box_width_ratio': 8.0,
            'l': 14.0,
            'num_groups': args.num_groups,
            'ground_side_len': 20.0,
            'pre_sample_box': True,
            'num_agents': args.num_agents,
        }
    else:
        print("Task name not supported")
        assert(False)

    reset_args = {
        'num_agents': args.num_agents,
    }
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
        'ordering_scheme' : args.ordering_scheme,
        #'add_prob' : args.add_prob,
        #'re_add_agents' : args.re_add_agents,
        'loss_type' : args.loss_type,
        'init_gap' : args.init_gap,
        'noise_scale' : args.noise_scale,
        'prog_feature': args.feature,
    }

    if  args.dropout_model != "random" and  not args.self_attn:
        env_args['comm_dropout_model'] = 'no_self'

    if args.dist_based_hard:
        env_args['comm_dropout_model'] = 'dist_based_hard'
        env_args['comm_deg'] = args.comm_deg

    exp_env = gym.make(args.env, args=env_args)

    input_dim = exp_env.n_features
    edge_dim = 2
    nhead = 1

    # Construct policy
    if args.policy == 'transformer_edge':
        policy_settings = {
            'transformer_model_dim' : args.transformer_model_dim,
            'nhead' : nhead,
            'transformer_fc_dim' : args.transformer_model_dim,
            'device' : device,
            'input_dim' : input_dim,
            'edge_dim' : edge_dim,
            'sep_self_edge' : False,
            'n_hops': args.hops,
            'with_sigma': args.alg == "pg",
            'sep_key_val': args.sep_key_val,
            'hard_attn' : False,
            'hard_attn_deg' : args.comm_deg,
            'rand_hard_attn' : False,
            'train_attention' : (not args.retrain_with_prog_attn and not args.retrain_with_dt_attn) or args.use_soft_with_prog,
            'only_obs_edge': args.only_obs_edge,
            'use_internal_state': args.use_internal_state,
            'num_out' : exp_env.nu,
            'use_sigmoid' : args.use_sigmoid
        }
        outputManager.say("\nPolicy settings:\n")
        outputManager.say(policy_settings)
        # initialize policy
        policy = neural_policy_bptt.TransformerEdgePolicy(policy_settings).to(device)

    elif args.policy == 'gcn':
        # yicheny TODO: edit
        policy_settings = {
            'hidden_dim' : 16,
            'device' : device,
            'input' : 'both',
        }
        outputManager.say("\nPolicy settings:\n")
        outputManager.say(policy_settings)
        # initialize policy
        policy = neural_policy.GCNPolicy(policy_settings).to(device)

    else:
        outputManager.say("Policy not supported")
        return

    if args.retrain_with_prog_attn:
        model_file = os.path.join(results_dir, 'model.pt')
        policy.load_state_dict(torch.load(model_file, map_location=device))

        progs = []
        if not args.sep_hops:
            prog_file = os.path.join(results_dir, "prog.txt"+args.suffix)

            file = open(prog_file)
            for l in file.readlines():
                prog = eval(l)
                progs.append(prog)
            file.close()
        else:
            for hop in range(args.hops):
                prog_file = os.path.join(results_dir, "prog.txt"+args.suffix +"_hop%i"%hop)
                file = open(prog_file)
                prog = eval(file.readlines()[0])
                progs.append(prog)
                file.close()

        for prog in progs:
            if args.self_attn:
                dt_file = os.path.join(results_dir, 'dt.joblib')
                dt_se = load(dt_file)
                prog.set_dt_self_edge(dt_se)
            else:
                prog.set_no_self()

            if args.only_obs_edge:
                prog.set_only_edge()

        policy.use_prog_for_attn(progs, use_soft_with_prog=args.use_soft_with_prog)

    if args.retrain_with_dt_attn:
        model_file = os.path.join(results_dir, 'model.pt')
        policy.load_state_dict(torch.load(model_file, map_location=device))

        dts = []
        for hop in range(args.hops):
            dt_file = os.path.join(results_dir, "dt_%ihop.joblib"%hop)
            dt = load(dt_file)
            dts.append(dt)
        policy.use_dt_for_attn(dts, use_soft_with_prog=args.use_soft_with_prog)


    model_save_file = os.path.join(results_dir, 'model%s.pt'%suffix)

    # Train policy
    if (args.retrain_with_prog_attn or args.retrain_with_dt_attn) and (not args.use_soft_with_prog):
        optimizer = torch.optim.Adam(policy.na_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    avg_cum_loss_hist = []
    avg_dist_loss_hist = []
    avg_num_col_hist = []
    for episode in tqdm(range(args.episode)):
        exp_env.set_reset_args(reset_args)
        state = exp_env.reset()
        cum_loss = 0.0
        cum_distance_loss = 0.0
        cum_num_collision = 0

        max_step = args.max_step

        discount = 1.0
        gamma = 1.0

        policy.reset_internal_state()

        if args.alg == 'bptt':
            for _ in range(max_step):
                if args.detach_state:
                    state = state.detach()
                obs = exp_env.get_obs(state)

                if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy):
                    relative_pos = exp_env.get_relative_pos(state)
                    comm_weights = exp_env.get_comm_weights(state)
                    if len(policy.attn_progs) > 0 or len(policy.attn_dts) > 0:
                        prog_input = exp_env.get_prog_features(state, relative_pos)
                    else:
                        prog_input = None
                    action = policy(obs, relative_pos, comm_weights, prog_input = prog_input)
                else:
                    action = policy(obs)

                state = exp_env.step1(state, action)

                total_loss, distance_loss, collision_loss = exp_env.get_loss(state)
                cum_distance_loss += distance_loss
                cum_num_collision += collision_loss

                cum_loss += discount * total_loss

                discount *= gamma

            optimizer.zero_grad()
            cum_loss.backward()
            optimizer.step()

        else:
            # policy gradient
            assert isinstance(policy, neural_policy.GCNPolicy)
            discount_pg = 0.7
            reward_history = []
            log_prob_history = None
            for _ in range(max_step):
                state = state.detach()
                # build graph
                g = putils.build_graph(state.data.cpu().numpy())
                g = g.to(device)

                obs = exp_env.get_obs(state)
                inputs = torch.cat((state, obs), 1) # (N, 4)

                mu, sigma_sq = policy(g, inputs)

                # Sample action
                action, log_prob = policy.sample_actions(mu, sigma_sq)

                if log_prob_history is None:
                    log_prob_history = log_prob
                else:
                    log_prob_history = torch.cat([log_prob_history, log_prob])

                # Step and get reward
                if args.reward == "change":
                    '''
                    last_state = state
                    state = exp_env.step1(state, action)
                    total_loss, distance_loss, collision_loss = exp_env.difference_loss(state, last_state)
                    '''
                    last_state = state
                    state = exp_env.step1(state, action)
                    total_loss_last, _, _ = exp_env.get_loss(last_state)
                    total_loss, distance_loss, collision_loss = exp_env.get_loss(state)
                    total_loss_diff = total_loss - total_loss_last
                    reward_history.append(-total_loss_diff)

                else:
                    state = exp_env.step1(state, action)
                    total_loss, distance_loss, collision_loss = exp_env.get_loss(state)
                    reward_history.append(-total_loss)

                cum_distance_loss += distance_loss
                cum_num_collision += collision_loss
                cum_loss += discount * total_loss
                discount *= gamma

            # cumulate and adjust reward
            rewards_adjusted = []
            R = 0
            for r in reward_history[::-1]:
                R = r + discount_pg * R
                rewards_adjusted.insert(0, R)
            rewards_adjusted = torch.FloatTensor(rewards_adjusted)
            rewards_adjusted = ((rewards_adjusted - rewards_adjusted.mean()) /
                (rewards_adjusted.std() + np.finfo(np.float32).eps))

            # compute loss and update
            loss = torch.sum(torch.mul(log_prob_history, Variable(rewards_adjusted).to(policy.device)).mul(-1), -1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        avg_dist_loss = cum_distance_loss.data.cpu().numpy() / max_step
        avg_dist_loss_hist.append(avg_dist_loss)
        avg_num_col = cum_num_collision.data.cpu().numpy() / max_step
        avg_num_col_hist.append(avg_num_col)
        avg_cum_loss = cum_loss.data.cpu().numpy() / max_step
        avg_cum_loss_hist.append(avg_cum_loss)

        if episode % 10 == 0:
            outputManager.say("Episode {}, average cumulated loss {:.2f}\t, average distance loss {:.2f}\t, average number of collisions {:.2f}".format(episode, avg_cum_loss, avg_dist_loss, avg_num_col))

        if episode % 50 == 0 or episode == args.episode - 1:
            torch.save(policy.state_dict(), model_save_file)

    filename = os.path.join(results_dir, "avg_cum_loss%s.txt"%suffix)
    np.savetxt(filename, avg_cum_loss_hist)

    filename = os.path.join(results_dir, "avg_num_col%s.txt"%suffix)
    np.savetxt(filename, avg_num_col_hist)

    filename = os.path.join(results_dir, "avg_dist_loss%s.txt"%suffix)
    np.savetxt(filename, avg_dist_loss_hist)

    filename = os.path.join(results_dir, "avg_cum_loss%s.png"%suffix)
    plot_stats_over_episodes(avg_cum_loss_hist, filename)

    filename = os.path.join(results_dir, "avg_num_col%s.png"%suffix)
    plot_stats_over_episodes(avg_num_col_hist, filename)

    filename = os.path.join(results_dir, "avg_dist_loss%s.png"%suffix)
    plot_stats_over_episodes(avg_dist_loss_hist, filename)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
