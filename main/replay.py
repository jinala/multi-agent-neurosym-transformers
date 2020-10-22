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
import pylab


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
    parser.add_argument('--use_internal_state', action='store_true', default=False,
        help='Use internal state in transformer')
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

    parser.add_argument('--suffix', type=str, default='',
        help='Suffix for saved program_policy')
    parser.add_argument('--setting_idx', type=str, default='',
        help='Setting idx')
    parser.add_argument('--with_sigma', action = 'store_true', default = False)
    parser.add_argument('--use_sigmoid', action = 'store_true', default = False)
    parser.add_argument('--hard_attn_deg', type=int, default=2, help='Hard attention degree')
    parser.add_argument('--use_soft_with_prog', action='store_true', default=False, help='Use soft attention together with prog_attn')

    parser.add_argument('--ordering_scheme', type = str , default = "min", help = "Ordering scheme for unassigned task, can be fixed or min")

    parser.add_argument('--sep_hops', action = "store_true", default = False, help='Learn prog for each hop separately')


    # traffic junction task params
    #parser.add_argument('--add_prob', type = float, default = 0.2)
    #parser.add_argument('--re_add_agents', action = "store_true", default = False)
    parser.add_argument('--loss_type', type = str, default = "both")
    parser.add_argument('--init_gap', type = int, default = 3)
    parser.add_argument('--feature', type=str, default='orig',
        help='Program features. Can be orig, quadratic, goal, both')

    parser.add_argument('--baseline', type = str, default="prog-retrained", help = "Replay baseline. Can be tf-full, hard, dist, prog, prog-retrained, dt, dt-retrained, det, det-retrained")

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

    if hasattr(saved_args, 'num_groups'):
        args.num_groups = saved_args.num_groups
    
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

    if hasattr(saved_args, 'ordering_scheme'):
        args.ordering_scheme = saved_args.ordering_scheme

    if hasattr(saved_args, 'init_gap'):
        args.init_gap = saved_args.init_gap
    if hasattr(saved_args, 'loss_type'):
        args.loss_type = saved_args.loss_type

    if hasattr(saved_args, 'task'):
        args.task = saved_args.task 
        if args.task == "random_cross" or args.task == "random_grid":
            args.env = "FormationTorch-v0"
        if args.task == "unassigned":
            args.task = "unlabeled"
        if args.task == "unlabeled":
            args.env =  "UnlabeledGoals-v0"

    if args.baseline == "prog-retrained":
        args.prog_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = True 
    elif args.baseline == "prog":
        args.prog_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = False
    elif args.baseline == "dt-retrained":
        args.dt_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = True
    elif args.baseline == "dt":
        args.dt_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = False
    elif args.baseline == "hard":
        args.hard_attn  = True 
        args.hard_attn_deg = 2 if args.task == "random_grid" else 5 
    elif args.baseline == "det-retrained":
        args.prog_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = True 
        args.suffix = "_no_random"
    elif args.baseline == "det":
        args.prog_attn = True 
        args.use_soft_with_prog = True 
        args.use_retrained_model = False
        args.suffix = "_no_random"
    elif args.baseline == "dist":
        args.dist_based_hard = True 
        args.comm_deg = 2 if args.task == "random_grid" else 5 
        args.sep_key_val = True if (args.task == "random_cross") else False


    if args.task == "unlabeled":
        args.sep_hops = True


    args.num_agents = saved_args.num_agents

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

    print(args)

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
            'l': 14.0,
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
        'ordering_scheme' : args.ordering_scheme,
        'init_gap' : args.init_gap,
        'loss_type' : args.loss_type,
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
    if args.baseline == "dist":
        filename = os.path.join(results_dir, "model_dist.pt")
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
            'rand_hard_attn' : args.rand_hard_attn,
            'train_attention' : True,
            'only_obs_edge' : args.only_obs_edge,
            'num_out' : exp_env.nu,
            'use_internal_state': args.use_internal_state,
            'use_sigmoid' : args.use_sigmoid,
            'hard_attn_deg' : args.hard_attn_deg,
        }
        # initialize policy
        policy = neural_policy_bptt.TransformerEdgePolicy(policy_settings).to(device)
        policy.load_state_dict(torch.load(filename , map_location=torch.device('cpu')))

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
    progs = []
    if args.prog_attn:
        progs = []
        if not args.sep_hops:
            prog_file = os.path.join(results_dir, "prog.txt"+args.suffix)

            file = open(prog_file)
            for l in file.readlines():
                prog = eval(l)
                #print(prog)
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

    if args.show_dt:
        prog_file = os.path.join(results_dir, "prog.txt"+args.suffix)
        file = open(prog_file)
        prog = eval(file.readlines()[0])




    # Get test settings
    test_settings = get_test_settings(args.task, args.num_groups)

    reset_args = {
        'num_agents': args.num_agents,
        #'setting': setting,
        #'seed': seed,
    }
    if args.setting_idx != "":
        reset_args['setting'] = test_settings[int(args.setting_idx)]
    # Save video
    my_animator = Animator(exp_env, policy, reset_args, args.show_dt, progs, args.feature)
    my_animator.draw_and_save()
    plt.show()

    

class Animator(object):
    """docstring for Animator."""

    def __init__(self, exp_env, policy, reset_args, show_dt, progs, feature):
        super(Animator, self).__init__()
        self.exp_env = exp_env
        self.policy = policy
        self.reset_args = reset_args
        self.show_dt = show_dt
        self.progs = progs
        self.feature = feature

        hops = self.policy.n_hops if hasattr(self.policy, "n_hops") else 0
        self.plot_num_col = 1 + hops + (1 if self.show_dt else 0)
        #if isinstance(self.policy, neural_policy.GCNPolicy):
        #    self.plot_num_col = 2

        self.num_collision = 0
        self.total_distance_loss = 0
        self.att_weights_episode = []

        # For attention of a specific agent
        self.sp = None
        self.sp1 = None

        self.rule_objs = []

    def draw_and_save(self):
        self.exp_env.set_reset_args(self.reset_args)
        self.state = self.exp_env.reset()
        #self.agent = 5# np.random.randint(0, len(self.exp_env.x) )
        #print("Agent: ", self.agent)

        self.max_step = self.exp_env._max_episode_steps
        self.num_collision = 0
        self.att_weights_episode = []

        self.fig, *self.plot_objects = self.exp_env.render(num_col = self.plot_num_col, save_video = False)

        #anim = FuncAnimation(self.fig, self.animate,
        #    frames=self.max_step, interval=10, blit=True)
        
        #anim.save('animate.mp4', codec='mpeg4')

        for i in range(self.max_step):
            self.animate(i)
            #if i%20 == 0:
            #    print("Col: ", self.get_num_collision())
            #    print("Dist: ", self.total_distance_loss)

    def plot_attn_choice(self, attn):
        fig = self.fig
        ax = fig.get_axes()[0]
        for obj in self.rule_objs:
            obj.remove()
        self.rule_objs = []
        agent = self.agent
        init = self.exp_env.x[agent]
        init = init.detach().cpu().numpy()
        scatter1 = ax.scatter([init[0]], [init[1]], c='m', s = 60)
        self.rule_objs.append(scatter1)
        _, selected_agents = torch.topk(attn[self.agent], 2)
        for a in selected_agents:
            pos = self.exp_env.x[a].detach().cpu().numpy()
            sc = ax.scatter([pos[0]], [pos[1]], c="k", s=60)
            self.rule_objs.append(sc)

    def plot_prog(self, num_cols):
        # For each rule, plot heatmap
        rng = 20
        x_lim = [-rng, rng]
        y_lim = [-rng, rng]

        agent = self.agent
        init = self.exp_env.x[agent]
        goal = self.exp_env.goals[agent]

        b1s = np.arange(x_lim[0], x_lim[1], 0.1)
        b2s = np.arange(y_lim[0], y_lim[1], 0.1)
        B2,B1 = pylab.meshgrid(b2s, b1s) # grid of point (N, N)
       
        N = B1.shape[0]
        prog_input = prep_prog_input(init, goal, B1, B2, feature= self.feature) # (N, N, d_f)
        #import pdb; pdb.set_trace()
        prog_input = prog_input.reshape(N*N, -1)

        prog = self.progs[0]
        i = 0
        fig = self.fig
        ax = fig.get_axes()[0]
        for obj in self.rule_objs:
            obj.remove()
        self.rule_objs = [] 
        colors = ['tab:gray', 'tab:brown', 'tab:olive','tab:pink']
        det_rules = [] 
        non_det_rules = []
        for rule in prog.rules:
            if isinstance(rule, program_attn_policy.DetAggRule):
                det_rules.append(rule)
            else:
                non_det_rules.append(rule)

        for rule in det_rules[0:0]:
            map_val = rule.eval_separate(prog_input) # (N*N,)
            map_val = map_val.reshape(N, N).numpy()
            val = np.transpose(map_val)

            norm = cm.colors.Normalize()

            im = ax.imshow(val, norm = norm, cmap="coolwarm", origin='lower', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], alpha=1.0)
            self.rule_objs.append(im)

        all_fv = torch.zeros(N, N, dtype = torch.float32)
        for rule in non_det_rules[1:2]:
            # Non-determinstic rule
            fv = rule.eval_separate(prog_input) # (N*N,)
            fv = fv.reshape(N, N)
            all_fv += fv 

        for i in range(1, len(non_det_rules)+1):
            indices = (all_fv == i).nonzero()
            x = indices[:, 0]/10.0 - rng
            y = indices[:, 1]/10.0- rng
                
            s = ax.scatter(x, y, alpha=0.01*i, c = 'tab:gray' )
            self.rule_objs.append(s)


        init = init.detach().cpu().numpy()
        goal = goal.detach().cpu().numpy()
        scatter1 = ax.scatter([init[0]], [init[1]], c='m', s = 60)
        scatter2 = ax.scatter([goal[0]], [goal[1]], marker="x", c='m', s = 60)
        self.rule_objs.append(scatter1)
        self.rule_objs.append(scatter2)
        selected_agents = (self.policy.prog_attn_wghts[0][self.agent] > 1e-2).nonzero()
        for a in selected_agents:
            pos = self.exp_env.x[a[0]].detach().cpu().numpy()
            sc = ax.scatter([pos[0]], [pos[1]], c="k", s=60)
            self.rule_objs.append(sc)



            

        


    def animate(self, i):
        

        obs = self.exp_env.get_obs(self.state)
        if isinstance(self.policy, neural_policy_bptt.TransformerEdgePolicy):
            relative_pos = self.exp_env.get_relative_pos(self.state)
            comm_weights = self.exp_env.get_comm_weights(self.state)
            prog_input = self.exp_env.get_prog_features(self.state, relative_pos)
            
            if not self.policy.with_sigma:
                action, att_weights = self.policy(obs, relative_pos, comm_weights, prog_input = prog_input, need_weights = True)
            else:
                action, _, att_weights = self.policy(obs, relative_pos, comm_weights, need_weights = True)

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

            #if i == 10:
            #if len(self.progs) > 0:
            #    self.plot_prog(self.plot_num_col)
            #    self.plot_attn_weights(self.policy.prog_attn_wghts, self.plot_num_col, start = 2)
            #else:
            #    self.plot_attn_choice(att_weights[1][0])
            self.plot_attn_weights(att_weights, self.plot_num_col, start = 2)

            self.att_weights_episode.append(att_weights[0].data.numpy())

            if self.show_dt:
                dt_att_weights = self.prog.eval(self.exp_env.get_prog_features(self.state))
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
        self.exp_env.render(action = None, num_col = self.plot_num_col, save_video = False)

        self.state = self.exp_env.step1(self.state, action)

        _, dist_loss, _ = self.exp_env.get_loss(self.state)

        self.num_collision += self.exp_env.check_collision1()
        self.total_distance_loss = dist_loss

        if self.plot_num_col == 1:
            return self.plot_objects
        else:
            return self.plot_objects #.append(self.im)

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

    def get_num_collision(self):
        return self.num_collision

    def get_att_weights(self):
        return np.concatenate(self.att_weights_episode, axis=0) # (max_step, h, N, N)

def prep_prog_input(init, goal, B1, B2, noise_scale=0.0, feature='orig'):
    state = torch.tensor(init, dtype=torch.float32) # (2,)
    goal = torch.tensor(goal, dtype=torch.float32)

    Z = 15.0
    Z_norm = 20.0

    input = state - goal #(2,)
    input = input.unsqueeze(0) # (1, 2)
    input_norm = input.norm(dim = -1, keepdim=True) # (1, 1)
    input_dir = input/input_norm
    input_x = input_dir[:, 0] # (1,)
    input_y = input_dir[:, 1] # (1,)
    input_ang = torch.atan2(input_y, input_x).reshape((1, 1))
    input1 = torch.cat((input/Z, input_norm/Z_norm, input_ang/3.14), axis=-1) # (1, 4)

    N = B1.shape[0]
    s3 = input1.repeat(1, N*N).reshape(N, N, -1) # (N, N, 4)

    B1 = torch.tensor(B1, dtype=torch.float32).unsqueeze(2) # (N, N, 1)
    B2 = torch.tensor(B2, dtype=torch.float32).unsqueeze(2) # (N, N, 1)
    other_pos = torch.cat((B1, B2), dim=2) # (N, N, 2)
    #print(other_pos)
    #assert(False)
    edge_input = state.unsqueeze(0).unsqueeze(0) - other_pos # (N, N, 2)
    if noise_scale > 0:
        # Add noise
        noise = torch.randn(edge_input.shape) * noise_scale
        edge_input = edge_input + noise

    edge_input_norm = edge_input.norm(dim = -1).reshape((N, N, 1)) + 1e-4
    edge_input_dir = edge_input/edge_input_norm
    edge_input_x = edge_input_dir[:, :, 0]
    edge_input_y = edge_input_dir[:, :, 1]
    edge_input_ang = torch.atan2(edge_input_y, edge_input_x).reshape((N, N, 1))
    edge_input1 = torch.cat((edge_input/Z, edge_input_norm/Z_norm, edge_input_ang/3.14), axis=-1) # (N, N, 4)

    X = torch.cat((s3, edge_input1), -1) #(N, N, 8)

    if feature == 'quadratic':
        X = get_quadratic(X)

    elif feature == 'goal':
        X = get_goal_rel(X)

    elif feature == 'both':
        X = get_quadratic(X)
        X = get_goal_rel(X)

    return X


def get_quadratic(X):
    x_self = X[:, :, 0:1]
    y_self = X[:, :, 1:2]
    x_other = X[:, :, 4:5]
    y_other = X[:, :, 5:6]
    X = torch.cat([X, x_self*x_other, x_self*y_other, y_self*x_other, y_self*y_other], -1)
    return X


def get_goal_rel(X):
    goal_norm = X[:, :, 0:2].norm(dim=-1, keepdim=True) + 1e-8
    goal_dir = - X[:, :, 0:2]/goal_norm
    goal_y_dir = torch.cat([-goal_dir[:, :, 1:2], goal_dir[:, :, 0:1]], -1)
    other_pos = - X[:, :, 4:6]
    rel_to_goal_x = torch.sum(other_pos * goal_dir, dim=-1, keepdim=True)
    rel_to_goal_y = torch.sum(other_pos * goal_y_dir, dim=-1, keepdim=True)
    rel_to_goal_ang = torch.atan2(rel_to_goal_y, rel_to_goal_x)
    rel_to_goal_ang_abs = torch.abs(rel_to_goal_ang)
    X = torch.cat([X, rel_to_goal_x, rel_to_goal_y, rel_to_goal_ang/3.14, rel_to_goal_ang_abs/3.14], -1)
    return X


if __name__ == '__main__':
    
    main()
    