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
import main.utils as utils

from joblib import dump, load
from main.policy.program_attn_policy import *
from main.policy import program_attn_policy


from tasks.task import *


def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to test')
    parser.add_argument('--result_dir', type=str, default='test', help='folder to store results')
    parser.add_argument('--save_dir', type=str,
        help='folder to store plotting results')

    parser.add_argument('--task', type=str, default="random_cross")
    parser.add_argument('--num_groups', type=int, default=3)
    parser.add_argument('--alg', type = str, default="bptt", help = "Reinforcement learning algorithm, can be bptt or pg")
    parser.add_argument('--show_dt', action = 'store_true', default=False, help="Show the attention learned by the decision tree")


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

    parser.add_argument('--env', type=str, default='cross_arrow',
        help='type of env, can be cross_arrow, two_groups')
    parser.add_argument('--sep_self_edge', action='store_true', default=False,
        help='Separate self edge embedding')
    parser.add_argument('--gap', type=float, default=3.0, help='Average gap for initial location, for two_groups crossing. Times the collision_dist.')
    parser.add_argument('--col_loss', type=str, default='ramp',
        help='type of collision loss, can be ramp, exp')
    parser.add_argument('--reward', type=str, default='total',
        help='reward structure. Can be total/change')
    parser.add_argument('--wall_height_ratio', type=float, default='0.6', help='Ratio of wall height to distance')
    parser.add_argument('--tunnel_height_ratio', type=float, default='3.0', help='Ratio of tunnel height to collision_dist')
    parser.add_argument('--clip_speed', action='store_true', default=False,
        help='Clipping actual speed')
    parser.add_argument('--load_all_args', action='store_true', default=False)

    parser.add_argument('--hard_attn', action = 'store_true', default=False, help="Use hard attention in transformer")
    parser.add_argument('--rand_hard_attn', action = 'store_true', default=False, help="Use rand hard attention in transformer")
    parser.add_argument('--prog_attn', action = 'store_true', default=False, help="Use prog attention in transformer")
    parser.add_argument('--dt_attn', action = 'store_true', default=False, help="Use dt attention in transformer")
    parser.add_argument('--use_retrained_model', action = 'store_true', default=False, help="Use retrained transformer model with prog attn")

    parser.add_argument('--ratio_to_attend', type=float, default='0.1', help='Ratio of agents to attend at each step')
    parser.add_argument('--inhibit_within_group_attn', action='store_true', default=False, help='inhibit within group attention when testing hard attention')

    parser.add_argument('--self_attn', action = 'store_true', default=False, help='To have self attention in transformer')
    parser.add_argument('--dist_based_hard', action = 'store_true', default=False, help='To use distance based degree constraint for transformer')
    parser.add_argument('--comm_deg', type=int, default=2, help='Distance based degree constraint for transformer')
    parser.add_argument('--only_obs_edge', action='store_true', default=False, help='To have transformer only use edge feature to decide who to attend to')
    parser.add_argument('--use_soft_with_prog', action='store_true', default=False, help='Use soft attention together with prog_attn')

    parser.add_argument('--suffix', type=str, default='',
        help='Suffix for saved program_policy')

    parser.add_argument('--with_sigma', action = 'store_true', default = False)

    parser.add_argument('--use_sigmoid', action = "store_true", default = False, help = "Sigmoid for action")
    parser.add_argument('--use_internal_state', action = 'store_true', default=False, help="Separate encodings for key and value in transformer")
    parser.add_argument('--grid_l', type=float, default=14.0,
        help='Grid cell size')

    parser.add_argument('--hard_attn_deg', type=int, default=2, help='Hard attention degree')

    parser.add_argument('--noise_scale', type=float, default=0.0,
        help='Noise scale for observations')

    parser.add_argument('--ordering_scheme', type = str , default = "min", help = "Ordering scheme for unassigned task, can be fixed or min")

    parser.add_argument('--sep_hops', action = "store_true", default = False, help='Learn prog for each hop separately')
    parser.add_argument('--init_gap', type = int, default = 0)
    parser.add_argument('--loss_type', type = str, default = "action", help = "can be both, action, dist")

    parser.add_argument('--use0123', action = "store_true", default = False)
    parser.add_argument('--useurl', action = "store_true", default = False)
    parser.add_argument('--plot_hard', action = "store_true", default = False)

    parser.add_argument('--feature', type=str, default='orig',
        help='Program features. Can be orig, quadratic, goal, both')

    return parser.parse_args()

program_attn_policy.dev = torch.device("cpu")

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
    if hasattr(saved_args, 'grid_l'):
        args.grid_l = saved_args.grid_l
    if hasattr(saved_args, 'ordering_scheme'):
        args.ordering_scheme = saved_args.ordering_scheme
    if hasattr(saved_args, 'init_gap'):
        args.init_gap = saved_args.init_gap
    if hasattr(saved_args, 'loss_type'):
        args.loss_type = saved_args.loss_type

    if hasattr(saved_args, 'noise_scale'):
        args.noise_scale = saved_args.noise_scale

    args.comm_deg = saved_args.comm_deg

    if args.load_all_args:
        args.num_agents = saved_args.num_agents
        args.env = saved_args.env
        args.hard_attn = saved_args.hard_attn

    return args


def main():
    # Set up parameters
    args = get_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.result_dir)

    device = torch.device("cpu")

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
    elif args.task == 'random_group':
        task_args = {
            'box_width_ratio': 8.0,
            'l': 14.0,
            'num_groups': args.num_groups,
            'ground_side_len': 20.0,
            'num_agents': args.num_agents,
        }
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
        'ordering_scheme' : args.ordering_scheme,
        'init_gap' : args.init_gap,
        'loss_type' : args.loss_type,
        'noise_scale': args.noise_scale,
        'prog_feature': args.feature,
    }

    if not args.self_attn:
        env_args['comm_dropout_model'] = 'no_self'

    if args.dist_based_hard:
        env_args['comm_dropout_model'] = 'dist_based_hard'
        env_args['comm_deg'] = args.comm_deg

    exp_env = gym.make(args.env, args=env_args)
    exp_env._max_episode_steps = args.max_step

    input_dim = exp_env.n_features
    edge_dim = 2
    nhead = 1

    # Load policy
    filename = os.path.join(results_dir, "model.pt")
    if args.use_retrained_model and args.prog_attn:
        suffix = args.suffix + "_retrained"
        filename = os.path.join(results_dir, "model%s.pt"%suffix)
    if args.use_retrained_model and args.dt_attn:
        suffix = args.suffix + "_dt_retrained"
        filename = os.path.join(results_dir, "model%s.pt"%suffix)
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
            'with_sigma' : args.with_sigma,
            'sep_key_val' : args.sep_key_val,
            'hard_attn' : args.hard_attn,
            'hard_attn_deg' : args.hard_attn_deg,
            'rand_hard_attn' : args.rand_hard_attn,
            'train_attention' : True,
            'only_obs_edge' : args.only_obs_edge,
            'num_out' : exp_env.nu,
            'use_internal_state': args.use_internal_state,
            'use_sigmoid' : args.use_sigmoid,
        }
        # initialize policy
        policy = neural_policy_bptt.TransformerEdgePolicy(policy_settings).to(device)
        policy.load_state_dict(torch.load(filename))

    elif args.policy == 'gcn':
        policy_settings = {
            'hidden_dim' : 16,
            'device' : device,
            'input' : 'both',
        }
        # initialize policy
        policy = neural_policy.GCNPolicy(policy_settings).to(device)

    else:
        print("Policy not supported")
        return

    policy.eval()

    if args.prog_attn:
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
    if args.dt_attn:
        dts = [] 
        for hop in range(args.hops):
            dt_file = os.path.join(results_dir, "dt_%ihop.joblib"%hop)
            dt = load(dt_file)
            dts.append(dt)
        policy.use_dt_for_attn(dts, use_soft_with_prog=args.use_soft_with_prog)

    # Get test settings
    test_settings = get_test_settings(args.task, args.num_groups)
    if args.use0123:
        test_settings = [[0,1,2,3]]
    elif args.useurl:
        test_settings = ['url']

    # For each settings, run different seeds and get results
    save_dir = os.path.join(results_dir, args.save_dir) 
    utils.create_dir(save_dir)

    for setting in test_settings:
        sub_dir_name = str(setting).strip('[]').replace(', ', '').replace('[', '').replace(']', '')
        sub_dir = os.path.join(save_dir, sub_dir_name)
        utils.create_dir(sub_dir)

        # stats to collect:
        # average cumulated loss, average number of collisions, average (over agents) degree in (average over steps), average (over agents) degree out (average over steps), max (over agents) degree in (average over steps), max (over agents) degree out (average over steps)
        if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy) or isinstance(policy, neural_policy.GCNPolicy):
            stats_item = ['avg_cum_loss', 'avg_distance_loss', 'avg_collision_loss', 'avg_num_col_within', 'avg_num_col_across', 'avg_deg_in', 'avg_deg_out', 'max_deg_in', 'max_deg_out']
        else:
            stats_item = ['avg_cum_loss', 'avg_num_col_within', 'avg_num_col_across']

        stats_dict = {}
        for item in stats_item:
            stats_dict[item+'_list'] = []

        for seed in range(args.num_seeds):
            reset_args = {
                'num_agents': args.num_agents,
                'setting': setting,
                'seed': seed,
            }

            if seed == 0:
                # Save video
                my_animator = Animator(exp_env, policy, sub_dir, reset_args, args.show_dt, args.plot_hard)
                my_animator.draw_and_save()

            stats = run_and_get_stats(exp_env, policy, reset_args)
            for item in stats_item:
                stats_dict[item+'_list'].append(stats[item])

        # Save stats
        for item in stats_item:
            stats_dict[item+'_mean'] = np.mean(stats_dict[item+'_list'])
            stats_dict[item+'_std'] = np.std(stats_dict[item+'_list'])

        stats_file = os.path.join(sub_dir, 'stats.json')
        if len(test_settings) > 1:
            with open(stats_file, 'w') as outf:
                json.dump(stats_dict, outf, indent=2)

    
    if len(test_settings) <= 1:
        # Get stats for random across settings
        np.random.seed(525)
        if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy) or isinstance(policy, neural_policy.GCNPolicy):
            stats_item = ['avg_cum_loss', 'avg_distance_loss', 'avg_collision_loss', 'avg_num_col_within', 'avg_num_col_across', 'avg_deg_in', 'avg_deg_out', 'max_deg_in', 'max_deg_out']
        else:
            stats_item = ['avg_cum_loss', 'avg_num_col_within', 'avg_num_col_across']

        stats_dict = {}
        for item in stats_item:
            stats_dict[item+'_list'] = []

        for _ in range(args.num_seeds*5):
            reset_args = {
                'num_agents': args.num_agents,
            }

            stats = run_and_get_stats(exp_env, policy, reset_args)
            for item in stats_item:
                stats_dict[item+'_list'].append(stats[item])

        # Save stats
        for item in stats_item:
            stats_dict[item+'_mean'] = np.mean(stats_dict[item+'_list'])
            stats_dict[item+'_std'] = np.std(stats_dict[item+'_list'])

        stats_file = os.path.join(save_dir, 'stats.json')
        with open(stats_file, 'w') as outf:
            json.dump(stats_dict, outf, indent=2)


def run_and_get_stats(exp_env, policy, reset_args):
    exp_env.set_reset_args(reset_args)
    state = exp_env.reset()
    policy.reset_internal_state()

    max_step = exp_env._max_episode_steps

    # Stats history
    cum_loss = 0.0
    cum_distance_loss = 0.0
    cum_collision_loss = 0.0
    total_num_col = np.array([0, 0])
    if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy) or isinstance(policy, neural_policy.GCNPolicy):
        avg_deg_in_hist = []
        avg_deg_out_hist = []
        max_deg_in_hist = []
        max_deg_out_hist = []

    for step in range(max_step):
        obs = exp_env.get_obs(state)
        if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy):
            relative_pos = exp_env.get_relative_pos(state)
            comm_weights = exp_env.get_comm_weights(state)
            prog_input = exp_env.get_prog_features(state, relative_pos)
            if not policy.with_sigma:
                action, att_weights = policy(obs, relative_pos, comm_weights, prog_input = prog_input, need_weights = True)
            else:
                action, _, att_weights = policy(obs, relative_pos, comm_weights, prog_input = prog_input, need_weights = True)

            # att_weights: (1, N, N)
            deg_in = 0
            deg_out = 0
            for i in range(len(att_weights)):
                comm_map = (att_weights[i][0] > 1e-6)
                deg_in = torch.sum(comm_map, axis=1).float()
                deg_out = torch.sum(comm_map, axis=0).float()
                avg_deg_in_hist.append(torch.mean(deg_in).tolist())
                avg_deg_out_hist.append(torch.mean(deg_out).tolist())
                max_deg_in_hist.append(torch.max(deg_in).tolist())
                max_deg_out_hist.append(torch.max(deg_out).tolist())

        elif isinstance(policy, neural_policy.GCNPolicy):
            g, deg_in, deg_out, _ = utils.build_graph(state.data.cpu().numpy(), get_degree=True)
            inputs = torch.cat((state, obs), 1) # (N, 4)
            mu, sigma_sq = policy(g, inputs)

            # Sample action
            action, _ = policy.sample_actions(mu, sigma_sq)

            avg_deg_in_hist.append(np.mean(deg_in).item())
            avg_deg_out_hist.append(np.mean(deg_out).item())
            max_deg_in_hist.append(np.max(deg_in).item())
            max_deg_out_hist.append(np.max(deg_out).item())

        else:
            action = policy(obs)

        state = exp_env.step1(state, action)

        total_loss, distance_loss, collision_loss = exp_env.get_loss(state)
        cum_loss += total_loss.tolist()
        cum_distance_loss += distance_loss.tolist()
        cum_collision_loss += collision_loss.tolist()
        total_num_col += exp_env.check_collision1()

        #if exp_env.done(state):
        #    break

    stats = {
        'avg_cum_loss': cum_loss/max_step,
        'avg_distance_loss' : cum_distance_loss/max_step,
        'avg_collision_loss' : cum_collision_loss/max_step,
        'avg_num_col_within': total_num_col[0]/max_step,
        'avg_num_col_across': total_num_col[1]/max_step,
    }
    if isinstance(policy, neural_policy_bptt.TransformerEdgePolicy) or isinstance(policy, neural_policy.GCNPolicy):
        stats['avg_deg_in'] = sum(avg_deg_in_hist)/len(avg_deg_in_hist)
        stats['avg_deg_out'] = sum(avg_deg_out_hist)/len(avg_deg_out_hist)
        stats['max_deg_in'] = sum(max_deg_in_hist)/len(max_deg_in_hist)
        stats['max_deg_out'] = sum(max_deg_out_hist)/len(max_deg_out_hist)

    return stats


class Animator(object):
    """docstring for Animator."""

    def __init__(self, exp_env, policy, save_dir, reset_args, show_dt, plot_hard=False, seed = -1):
        super(Animator, self).__init__()
        self.exp_env = exp_env
        self.policy = policy
        self.save_dir = save_dir
        self.reset_args = reset_args
        self.show_dt = show_dt
        self.seed = seed

        hops = self.policy.n_hops if hasattr(self.policy, "n_hops") else 0
        self.plot_num_col = 1 + hops + (1 if self.show_dt else 0)
        #if isinstance(self.policy, neural_policy.GCNPolicy):
        #   self.plot_num_col = 2

        self.num_collision = 0
        self.att_weights_episode = []

        # For attention of a specific agent
        self.sp = None
        self.sp1 = None
        self.plot_hard = plot_hard

    def draw_and_save(self):
        self.exp_env.set_reset_args(self.reset_args)
        self.state = self.exp_env.reset()
        self.policy.reset_internal_state()

        self.max_step = self.exp_env._max_episode_steps
        self.num_collision = 0
        self.att_weights_episode = []

        self.fig, self.scatter1 = self.exp_env.render(num_col = self.plot_num_col, save_video = True)

        anim = FuncAnimation(self.fig, self.animate,
            frames=self.max_step, interval=100, blit=True)
        suffix = ""
        if self.seed >= 0:
            suffix = "_%i"%(self.seed)
        anim.save(os.path.join(self.save_dir, 'animate%s.mp4'%suffix), codec='mpeg4', bitrate=5000)

        num_col = self.get_num_collision()

        #stats_file = os.path.join(self.save_dir, 'stats{}.txt'.format(mode))
        #with open(stats_file, "w") as fout:
        #    fout.write("Total number of collisions: {}".format(num_col))

        if len(self.att_weights_episode) > 0:
            att_weights = self.get_att_weights()
            att_weights_file = os.path.join(self.save_dir, 'wgts.npy')
            np.save(att_weights_file, att_weights)

    def animate(self, i):
        self.exp_env.render(num_col = self.plot_num_col, save_video = True)

        obs = self.exp_env.get_obs(self.state)
        if isinstance(self.policy, neural_policy_bptt.TransformerEdgePolicy):
            relative_pos = self.exp_env.get_relative_pos(self.state)
            comm_weights = self.exp_env.get_comm_weights(self.state)
            prog_input = self.exp_env.get_prog_features(self.state, relative_pos)
            if not self.policy.with_sigma:
                action, att_weights = self.policy(obs, relative_pos, comm_weights, prog_input = prog_input, need_weights = True)
            else:
                action, _, att_weights = self.policy(obs, relative_pos, comm_weights, prog_input = prog_input, need_weights = True)

            if False:
                # Plot attention for a specific agent
                i1 = 6
                weights = att_weights[0][0][i1].detach().numpy()
                marker_sizes = weights*50.0
                ss = self.state.detach().numpy()
                fig = self.fig
                ax = fig.get_axes()[0]
                if i == 0:
                    self.sp1 = ax.scatter([ss[i1][0]], [ss[i1][1]], c='m', s = 60)
                    self.sp = ax.scatter(ss[:,0], ss[:, 1], facecolors='none', edgecolors='k', s = marker_sizes)
                else:
                    X = ss[:,0]
                    Y = ss[:,1]
                    self.sp.set_offsets(np.c_[X, Y])
                    self.sp.set_sizes(marker_sizes)

                    self.sp1.set_offsets([ss[i1]])

            if self.plot_hard:
                self.plot_attn_weights([self.policy.prog_attn_wghts], self.plot_num_col, start = 2)
            else:
                self.plot_attn_weights(att_weights, self.plot_num_col, start = 2)

            self.att_weights_episode.append(att_weights[0].data.numpy())

            if self.show_dt:
                dt_att_weights = self.policy.compute_dt_att_weights(dt, obs, relative_pos)
                self.plot_attn_weights(dt_att_weights, self.plot_num_col, start = 3)

        elif isinstance(self.policy, neural_policy.GCNPolicy):
            g, _, _, adj_matrix = utils.build_graph(self.state.data.cpu().numpy(), get_degree=True)
            inputs = torch.cat((self.state, obs), 1) # (N, 4)
            mu, sigma_sq = self.policy(g, inputs)

            # Sample action
            action, _ = self.policy.sample_actions(mu, sigma_sq)

            # Plot communication
            adj_matrix = np.float32(adj_matrix)
            self.plot_comm_graph(np.transpose(adj_matrix), self.plot_num_col)

        else:
            action = self.policy(obs)

        self.state = self.exp_env.step1(self.state, action)

        self.num_collision += self.exp_env.check_collision1()

        if self.plot_num_col == 1:
            return self.scatter1,
        else:
            return self.scatter1, self.im,

        #if self.exp_env.done(self.state):
        #    break

    def plot_comm_graph(self, comm_matrix, num_col):
        ctr = 2
        fig = self.fig
        norm = cm.colors.Normalize()
        allaxes = fig.get_axes()
        if len(allaxes) < ctr:
            ax = fig.add_subplot(1, num_col, ctr)
        else:
            ax = allaxes[ctr - 1]
        ax.cla()
        self.im = ax.imshow(comm_matrix, norm = norm, cmap="coolwarm")

    def plot_attn_weights(self, att_weights, num_col, start = 2):
        ctr = start
        fig = self.fig
        norm = cm.colors.Normalize()

        allaxes = fig.get_axes()

        for att in att_weights:
            att = att.squeeze(0).squeeze(0)
            att = att.data.numpy()
            if len(allaxes) < ctr:
                ax = fig.add_subplot(1, num_col, ctr)
            else:
                ax = allaxes[ctr - 1]
            ax.cla()
            ctr += 1
            self.im = ax.imshow(att, norm = norm, cmap="coolwarm")
            ax.set_title("Round %i"%(ctr - 2))
            #fig.colorbar(self.im, ax=ax)

    def get_num_collision(self):
        return self.num_collision

    def get_att_weights(self):
        return np.concatenate(self.att_weights_episode, axis=0) # (max_step, h, N, N)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
