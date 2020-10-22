import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
import itertools
import random
import pdb
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class FormationFlyingTorchEnv(gym.Env):

    def __init__(self, args=None):
        '''
        args = {
            'task',
            'device',
            'num_agents'=3,
            'collision_penalty'=0.0,
            'collision_dist'=0.1,
            'comm_dropout_model'=None,
            'comm_dropout_percent'=0,
            'col_loss'='ramp',
            'collision_penalty_inter_g'=5*collision_penalty,
            'collision_dist_inter_g'=2.0,
            'clip_speed'=False,
        }
        '''

        assert args, "Need to specify args for FormationFlyingTorch1Env"
        assert 'task' in args, "Need to specify task in args for FormationFlyingTorch1Env"
        # Unpack args
        self.task = args['task']
        self.n_agents = args['num_agents'] if 'num_agents' in args else 3
        self.collision_penalty = args['collision_penalty'] if 'collision_penalty' in args else 0.0
        self.collision_dist = args['collision_dist'] if 'collision_dist' in args else 0.1
        self.device = args['device']
        self.comm_dropout_model = args['comm_dropout_model'] if 'comm_dropout_model' in args else None
        self.comm_dropout_percent = args['comm_dropout_percent'] if 'comm_dropout_percent' in args else 0
        self.col_loss = args['col_loss'] if 'col_loss' in args else 'ramp'
        self.collision_penalty_inter_g = args['collision_penalty_inter_g'] if 'collision_penalty_inter_g' in args else 5*self.collision_penalty
        self.collision_dist_inter_g = args['collision_dist_inter_g'] if 'collision_dist_inter_g' in args else 2.0
        self.clip_speed = args['clip_speed'] if 'clip_speed' in args else False
        self.comm_deg = args['comm_deg'] if 'comm_deg' in args else 2

        self.noise_scale = args['noise_scale'] if 'noise_scale' in args else 0.0

        self.prog_feature = args['prog_feature'] if 'prog_feature' in args else 'orig' # orig, quadratic, goal, both

        # problem parameters from file
        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel'])

        # number states per agent
        self.nx_system = 2
        # number of actions per agent
        self.nu = 2

        self.n_features = 2

        if self.prog_feature == 'orig':
            self.n_prog_features = 8
        elif self.prog_feature == 'quadratic' or self.prog_feature == 'goal':
            self.n_prog_features = 12
        elif self.prog_feature == 'both':
            self.n_prog_features = 16
        else:
            raise Error("Type of program feature not supported")

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        '''
        # TODO : what should the action space be? is [-1,1] OK?
        self.action_space = spaces.Box(low=-self.v_max, high=self.v_max, shape=(self.nu*self.n_agents,),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)
        '''

        self.fig = None
        self.scatter1 = None
        self.seed()

        self.sep_col_metric = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def preprocess_action(self, action):
        if self.clip_speed:
            speed = torch.norm(action, dim=-1, keepdim=True) + 1e-6 # (N, 1)
            action = action / speed * torch.clamp(speed, 0, self.v_max)
        else:
            action = torch.clamp(action, -self.v_max, self.v_max)
        return action

    def step1(self, state, action):

        action = self.preprocess_action(action)
        new_state = state + action*self.dt
        self.x = new_state.detach()
        return new_state

    def done(self, x):
        diffs_x = torch.abs(x[:,0] - self.goal_xpoints)
        diffs_y = torch.abs(x[:,1] - self.goal_ypoints)

        return (diffs_x < 0.2).all() and (diffs_y < 0.2).all()


    def get_obs(self, x):
        return x - self.goals

    def get_prog_features(self, state, relative_pos):
        input = self.get_obs(state)
        edge_input = relative_pos

        if input.dim() == 2:
            input = input.unsqueeze(0)
        if edge_input.dim() == 2:
            edge_input = edge_input.unsqueeze(0)

        S, N, d_in = input.shape
        _, _, d_in_edge = edge_input.shape

        Z = 15.0
        Z_norm = 20.0

        input_norm = input.norm(dim = -1).reshape((S, N, 1)) + 1e-4
        input_dir = input/input_norm
        input_x = input_dir[:, :, 0]
        input_y = input_dir[:, :, 1]
        input_ang = torch.atan2(input_y, input_x).reshape((S, N, 1))
        input1 = torch.cat((input/Z, input_norm/Z_norm, input_ang/3.14), axis=-1) # dir, norm/Z
        w = d_in + 2
        s1 = input1.repeat(1, 1, N).reshape((S, N, N, w))

        if True: #self.only_obs_edge:
            s3 = s1
        else:
            s2 = input1.repeat(1, N, 1).reshape((S, N, N, w))
            s3 = torch.cat((s1, s2), -1) # (N, N, w + w)

        edge_input_norm = edge_input.norm(dim = -1).reshape((S, N*N, 1)) + 1e-4
        edge_input_dir = edge_input/edge_input_norm
        edge_input_x = edge_input_dir[:, :, 0]
        edge_input_y = edge_input_dir[:, :, 1]
        edge_input_ang = torch.atan2(edge_input_y, edge_input_x).reshape((S, N*N, 1))
        edge_input1 = torch.cat((edge_input/Z, edge_input_norm/Z_norm, edge_input_ang/3.14), axis=-1) # dir, norm/Z
        ew = d_in_edge +  2
        X = torch.cat((s3, edge_input1.reshape((S, N, N, ew))), -1) #(S, N, N , w+w+ew)

        if self.prog_feature == 'quadratic':
            X = self.get_quadratic(X)

        elif self.prog_feature == 'goal':
            X = self.get_goal_rel(X)

        elif self.prog_feature == 'both':
            X = self.get_quadratic(X)
            X = self.get_goal_rel(X)

        return X

    def get_quadratic(self, X):
        x_self = X[:, :, :, 0:1]
        y_self = X[:, :, :, 1:2]
        x_other = X[:, :, :, 4:5]
        y_other = X[:, :, :, 5:6]
        X = torch.cat([X, x_self*x_other, x_self*y_other, y_self*x_other, y_self*y_other], -1)
        return X

    def get_goal_rel(self, X):
        goal_norm = X[:, :, :, 0:2].norm(dim=-1, keepdim=True) + 1e-8
        goal_dir = - X[:, :, :, 0:2]/goal_norm
        goal_y_dir = torch.cat([-goal_dir[:, :, :, 1:2], goal_dir[:, :, :, 0:1]], -1)
        other_pos = - X[:, :, :, 4:6]
        rel_to_goal_x = torch.sum(other_pos * goal_dir, dim=-1, keepdim=True)
        rel_to_goal_y = torch.sum(other_pos * goal_y_dir, dim=-1, keepdim=True)
        rel_to_goal_ang = torch.atan2(rel_to_goal_y, rel_to_goal_x)
        rel_to_goal_ang_abs = torch.abs(rel_to_goal_ang)
        X = torch.cat([X, rel_to_goal_x, rel_to_goal_y, rel_to_goal_ang/3.14, rel_to_goal_ang_abs/3.14], -1)
        return X

    # state: torch.Tensor (N, 2)
    # Return: torch.Tensor (N^2, 2)
    def get_relative_pos(self, state):
        state_dim = state.size()[-1]
        relative_pos = state.unsqueeze(1) - state # (N, N, 2)
        relative_pos = relative_pos.contiguous().view(-1, state_dim) # (N^2, 2)
        if self.noise_scale > 0:
            # Add noise
            noise = torch.randn(relative_pos.shape, device=self.device) * self.noise_scale
            relative_pos = relative_pos + noise
        return relative_pos

    # state : torch.Tensor (N, 2)
    # Return: torch.Tensor(NxN)
    # weight = 0: no communication
    # weight = 100: full communication (no dropout)
    def get_comm_weights(self, state):
        N = state.size()[0]
        if self.comm_dropout_model == None:
            return torch.ones((N, N)).to(self.device)*100
        elif self.comm_dropout_model == "random":
            res = torch.ones((N, N)).to(self.device)*(100 - self.comm_dropout_percent)
            res.fill_diagonal_(0)
            return res
        elif self.comm_dropout_model == "dist_based":
            relative_pos = state.unsqueeze(1) - state # (N, N, 2)
            ds = torch.norm(relative_pos, dim=2) # (N, N)
            res = torch.ones((N, N)).to(self.device)
            res[ds <= 1.0] = 100
            res[ds > 1.0] = (100 - self.comm_dropout_percent)
            #weights = 100*torch.exp(-ds/1.0)
            return res
        elif self.comm_dropout_model == "no_self":
            res = torch.ones((N, N)).to(self.device)*100
            res.fill_diagonal_(0)
            return res
        elif self.comm_dropout_model == "dist_based_hard":
            relative_pos = state.unsqueeze(1) - state # (N, N, 2)
            ds = torch.norm(relative_pos, dim=2) # (N, N)
            ds.fill_diagonal_(100000)
            res = torch.zeros((N, N)).to(self.device)
            _, smallest = torch.topk(ds, self.comm_deg, dim=1, largest=False)
            res.scatter_(1, smallest, 100)
            return res

    def get_loss(self, x):
        z = len(x)/self.n_agents
        distance_loss = torch.sum(torch.norm(self.goals - x, dim=1))#/z
        collision_loss = (self.check_collision(x) + self.obs_cost(x)*self.collision_penalty) #/z

        loss = distance_loss + collision_loss
        return loss, distance_loss, collision_loss

    def difference_loss(self, state, last_state):
        # Difference loss
        distance = torch.sum(torch.norm(self.goals - state, dim=1))
        distance_last = torch.sum(torch.norm(self.goals - last_state, dim=1))
        distance_loss = distance - distance_last

        # Collision loss
        collision_loss = self.check_collision(state) + self.obs_cost(state)*self.collision_penalty

        total_loss = distance_loss + collision_loss
        return total_loss, distance_loss, collision_loss

    def check_collision(self, state):
        if (not self.sep_col_metric) or (len(state) == self.n_agents):
            relative_pos = state.unsqueeze(1) - state # (N, N, 2)
            ds = torch.norm(relative_pos, dim=2) # (N, N)
            col_dist = self.collision_dist
            col_penalty = self.collision_penalty
            if self.col_loss == 'ramp':
                loss = torch.clamp(ds * (- col_penalty/col_dist) + 2 * col_penalty, min=0.0)
            elif self.col_loss == 'exp':
                loss = torch.exp(- 1/col_dist * (ds - col_dist)) * col_penalty
            else:
                raise Error("Type of collision loss not supported")
            loss.fill_diagonal_(0)
            collision_loss = torch.sum(loss) / 2
            return collision_loss
        else:
            groups = torch.split(state, self.n_agents, dim=0)
            total_loss = 0
            # intra group loss
            for b1 in groups:
                relative_pos = b1.unsqueeze(1) - b1 # (N, N, 2)
                ds = torch.norm(relative_pos, dim=2) # (N, N)
                col_dist = self.collision_dist
                col_penalty = self.collision_penalty
                if self.col_loss == 'ramp':
                    loss = torch.clamp(ds * (- col_penalty/col_dist) + 2 * col_penalty, min=0.0)
                elif self.col_loss == 'exp':
                    loss = torch.exp(- 1/col_dist * (ds - col_dist)) * col_penalty
                else:
                    raise Error("Type of collision loss not supported")
                loss.fill_diagonal_(0)
                total_loss = total_loss + torch.sum(loss) / 2

            # inter group loss
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    b1 = groups[i]
                    b2 = groups[j]
                    relative_pos = b1.unsqueeze(1) - b2 # (N, N, 2)
                    ds = torch.norm(relative_pos, dim=2) # (N, N)
                    col_dist = self.collision_dist_inter_g
                    col_penalty = self.collision_penalty_inter_g
                    if self.col_loss == 'ramp':
                        loss = torch.clamp(ds * (- col_penalty/col_dist) + 2 * col_penalty, min=0.0)
                    elif self.col_loss == 'exp':
                        loss = torch.exp(- 1/col_dist * (ds - col_dist)) * col_penalty
                    else:
                        raise Error("Type of collision loss not supported")
                    total_loss = total_loss + torch.sum(loss)

            return total_loss

    def check_collision1(self):
        num_collision_across = 0
        num_collision_within = 0
        n = self.n_agents if self.sep_col_metric else len(self.x)
        groups = torch.split(self.x, n, dim=0)
        for b1 in groups:
            robot_xs = b1[:,0]
            robot_ys = b1[:,1]
            n_agents = len(b1)
            for i in range(n_agents-1):
                for j in range(i+1, n_agents):
                    d = ((robot_xs[i] - robot_xs[j])**2 + (robot_ys[i] - robot_ys[j])**2)**0.5
                    if d <= self.collision_dist:
                        num_collision_within += 1


        for k1 in range(len(groups)):
            b1 = groups[k1]
            robot_xs_b1 = b1[:, 0]
            robot_ys_b1 = b1[:, 1]
            for k2 in range(k1+1, len(groups)):
                b2 = groups[k2]
                robot_xs_b2 = b2[:, 0]
                robot_ys_b2 = b2[:, 1]

                for i in range(len(b1)):
                    for j in range(len(b2)):
                        d = ((robot_xs_b1[i] - robot_xs_b2[j])**2 + (robot_ys_b1[i] - robot_ys_b2[j])**2)**0.5
                        if d <= self.collision_dist_inter_g:
                            num_collision_across += 1

        return np.array([num_collision_within, num_collision_across])


    def check_collision_walls(self, x):
        distances = self.get_coll_metric_walls(x)
        return torch.sum(distances) * self.collision_penalty

    def get_coll_metric_walls(self, x):
        dists = [self.walls(x[i]) for i in range(len(x))]
        dists = torch.tensor(dists).to(self.device)
        return torch.exp(-10.0*(dists - 0.1))

    def set_reset_args(self, reset_args):
        self.reset_args = reset_args

    def reset(self):
        assert self.reset_args, "Need to call set_reset_args before reset"
        init_pos, goals, obs_cost = self.task.sample_task(self.reset_args)
        init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)
        self.goals = torch.tensor(goals, dtype=torch.float32).to(self.device)
        self.obs_cost = obs_cost
        self.start_xpoints = init_pos[:, 0]
        self.start_ypoints = init_pos[:, 1]
        self.goal_xpoints = self.goals[:, 0]
        self.goal_ypoints = self.goals[:, 1]

        x = init_pos
        self.x = x

        self.fig = None

        return x


    def reset_with_pos(self, init_pos, goal_pos):
        _, _, obs_cost = self.task.sample_task(self.reset_args)
        goals = goal_pos

        init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)
        self.goals = torch.tensor(goals, dtype=torch.float32).to(self.device)
        self.obs_cost = obs_cost
        self.start_xpoints = init_pos[:, 0]
        self.start_ypoints = init_pos[:, 1]
        self.goal_xpoints = self.goals[:, 0]
        self.goal_ypoints = self.goals[:, 1]

        x = init_pos
        self.x = x

        self.fig = None

        return x

    def render(self, mode='human', action = None, num_col = 1, save_video = False):
        """
        Render the environment with agents as points in 2D space
        """
        xmin = min(min(self.start_xpoints), min(self.goal_xpoints)) - 10.0
        xmax = max(max(self.start_xpoints), max(self.goal_xpoints)) + 10.0
        ymin = min(min(self.start_ypoints), min(self.goal_ypoints)) - 10.0
        ymax = max(max(self.start_ypoints), max(self.goal_ypoints)) + 10.0

        if self.fig is None:
            if not save_video:
                plt.ion()
            fig = plt.figure(figsize = (5*num_col, 5))
            def handle_close(evt):
                exit()

            fig.canvas.mpl_connect('close_event', handle_close)
            if not save_video:
                plt.show()

            ax = fig.add_subplot(1, num_col, 1)

            colors =  self.task.robot_colors()# cm.rainbow(np.linspace(0, 1, len(self.x[:, 0])))
            scatter1 = ax.scatter(self.x[:, 0], self.x[:, 1], c=colors)
            scatter2 = ax.scatter(self.goal_xpoints, self.goal_ypoints, c=colors, marker="x")

            plt.title('%d Robots Formation'%len(self.x))
            #plt.gca().legend(('Robots'))

            self.task.plot()

            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            self.fig = fig
            self.scatter1 = scatter1
            self.scatter2 = scatter2

        X = self.x[:, 0]
        Y = self.x[:, 1]

        self.scatter1.set_offsets(np.c_[X, Y])


        self.fig.canvas.draw()
        if not save_video:
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        return self.fig, self.scatter1

    def close(self):
        pass
