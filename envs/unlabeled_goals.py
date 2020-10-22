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
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class UnlabeledGoalsEnv(gym.Env):

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
        self.ordering_scheme = args['ordering_scheme'] if 'ordering_scheme' in args else "min"
        self.loss_type = args['loss_type'] if 'loss_type' in args else "action"

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
        self.nu = self.n_agents


        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        self.num_nearest_goals = self.n_agents

        self.n_features = 2 + self.num_nearest_goals*2 #+ self.num_nearest_goals

        self.n_prog_features = self.n_features + 2 
        self.softmax = nn.Softmax(dim = -1)

        self.init_ordering = None
        self.prev_action = torch.zeros(self.n_agents, self.n_agents, dtype = torch.float32).to(self.device)

        self.id = torch.eye(self.n_agents, self.n_agents, dtype = torch.float32).to(self.device)



        '''
        # TODO : what should the action space be? is [-1,1] OK?
        self.action_space = spaces.Box(low=-self.v_max, high=self.v_max, shape=(self.nu*self.n_agents,),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)
        '''

        self.fig = None
        self.scatter1 = None
        self.seed()

        self.step = 0

        self.sep_col_metric = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def clamp_action(self, action):
        if self.clip_speed:
            speed = torch.norm(action, dim=-1, keepdim=True) + 1e-6 # (N, 1)
            action = action / speed * torch.clamp(speed, 0, self.v_max)
            #speed = torch.norm(action, dim=-1, keepdim=True) + 1e-6
            #print(speed)
        else:
            action = torch.clamp(action, -self.v_max, self.v_max)
        return action

    def preprocess_action(self, action):
        action = self.softmax(action)
        #new_action = torch.zeros(action.shape, dtype = torch.float32).to(self.device)
        #_, max_indices = torch.max(action, dim = -1)
        #new_action[np.arange(len(action)), max_indices.view(-1)] = 1
        return action

    def step1(self, state, action):
        action = self.softmax(action)
        max_per_goal, _ = torch.max(action, dim = 0)
        #print("Action: ", action)
        if self.prev_action != None:
            da = action - self.prev_action
            diff = torch.mul(da, da).sum()
        else:
            diff = 0
        #print(diff)
        self.prev_action = action 
        if self.step < 10:
            act_cost = 0.1
        else:
            act_cost = 1.0
        self.step += 1
        self.action_loss = -max_per_goal.sum()# + diff*act_cost

        rel_to_goals = state.unsqueeze(1) - self.goals
        dists = rel_to_goals.norm(dim = 2, keepdim=True)
        dir_to_goals = rel_to_goals/ dists
        #print(dir_to_goals) 

        combined_dirs = (dir_to_goals * action.unsqueeze(2)).sum(axis = 1)
        #print(combined_dirs)

        action = self.clamp_action(-combined_dirs*self.v_max)
        
        #action = self.clamp_action(action)
        #print(action)
        new_state = state + action*self.dt
        self.x = new_state.detach()
        return new_state

    def done(self, x):
        return False
        diffs_x = torch.abs(x[:,0] - self.goal_xpoints)
        diffs_y = torch.abs(x[:,1] - self.goal_ypoints)

        return (diffs_x < 0.2).all() and (diffs_y < 0.2).all()


    def get_obs(self, x):
        N = len(x)

        rel_to_goals = x.unsqueeze(1) - self.goals
        #dists = torch.norm(rel_to_goals, dim = 2)
        #_, smallest = torch.topk(dists, self.num_nearest_goals, dim=1, largest=False)

        #nearest_goals = rel_to_goals[np.arange(N).repeat(self.num_nearest_goals), smallest.view(-1)].view(N, self.num_nearest_goals*2)
        if self.ordering_scheme == "fixed":
            ordered_goals = rel_to_goals.view(N, self.num_nearest_goals*2)
        else:
            ordered_goals = rel_to_goals[np.arange(N).repeat(self.num_nearest_goals), self.init_ordering.view(-1)].view(N, self.num_nearest_goals*2)
        obs = torch.cat((x, ordered_goals), dim = 1) 
        #print("Obs: ", obs)
        return obs 

    def convert(self, x):
        P, Q = x.shape
        x_v = x.view(-1, 2)
        x_d = x_v.norm(dim = 1)/20.0
        x_x = x_v[:, 0] + 1e-4
        x_y = x_v[:, 1]
        x_ang = torch.atan(x_y/x_x)/3.14
        new_x = torch.cat((x_d.unsqueeze(1), x_ang.unsqueeze(1)), dim = 1)
        return new_x.view(P, Q)

    def get_prog_features(self, state, rel_pos):
        # state: (Nx2)
        assert(state.dim() == 2)
        input = self.get_obs(state) # N x(2N + 2)
        edge_input = rel_pos # N^2, 2

        if input.dim() == 2:
            input = input.unsqueeze(0)
        if edge_input.dim() == 2:
            edge_input = edge_input.unsqueeze(0)

        S, N, d_in = input.shape
        _, _, d_in_edge = edge_input.shape

        w = d_in 
        s1 = input.repeat(1, 1, N).reshape((S, N, N, w))

        ew = d_in_edge 
        X = torch.cat((s1, edge_input.reshape((S, N, N, ew))), -1)
        #print(X)
        return X 



    # state: torch.Tensor (N, 2)
    # Return: torch.Tensor (N^2, 2)
    def get_relative_pos(self, state):
        state_dim = state.size()[-1]
        relative_pos = state.unsqueeze(1) - state # (N, N, 2)
        relative_pos = relative_pos.contiguous().view(-1, state_dim) # (N^2, 2)
        #print("relative_pos", relative_pos)
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
        N = len(x)

        goals_to_agents = self.goals.unsqueeze(1) - x 
        dists = torch.norm(goals_to_agents, dim = 2)
        min_per_goal, _ = torch.min(dists, dim = 1)
        #min_dists, _ = torch.min(dists, dim = 1)
        #distance_loss = min_dists.sum()
        #distance_loss = (self.softmax(-dists)*dists).sum()
        if self.loss_type == "both":
            distance_loss = self.action_loss + min_per_goal.sum()
        elif self.loss_type == "dist":
            distance_loss = min_per_goal.sum()
        else:
            distance_loss = self.action_loss
        
        collision_loss = self.check_collision(x) #+ self.obs_cost(x)*self.collision_penalty

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
        init_pos, goals, _ = self.task.sample_task(self.reset_args)
        init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)
        self.goals = torch.tensor(goals, dtype=torch.float32).to(self.device)
        self.start_xpoints = init_pos[:, 0]
        self.start_ypoints = init_pos[:, 1]
        self.goal_xpoints = self.goals[:, 0]
        self.goal_ypoints = self.goals[:, 1]

        x = init_pos
        N = len(x)
        if self.ordering_scheme == "fixed":
            self.init_ordering = torch.arange(0, self.num_nearest_goals, 1).repeat(N).view(N, self.num_nearest_goals)
        else:
            rel_to_goals = x.unsqueeze(1) - self.goals
            dists = torch.norm(rel_to_goals, dim = 2)
            _, smallest = torch.topk(dists, self.num_nearest_goals, dim=1, largest=False)
            self.init_ordering = smallest
        #print(self.init_ordering)
        

        self.x = x

        self.fig = None
        self.arrows=[]

        self.prev_action = torch.zeros(self.n_agents, self.n_agents, dtype = torch.float32).to(self.device)

        return x

    def reset1(self, init_pos, goals):
        assert self.reset_args, "Need to call set_reset_args before reset"
        self.task.num_robots = self.reset_args['num_agents']
        init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)
        self.goals = torch.tensor(goals, dtype=torch.float32).to(self.device)
        self.start_xpoints = init_pos[:, 0]
        self.start_ypoints = init_pos[:, 1]
        self.goal_xpoints = self.goals[:, 0]
        self.goal_ypoints = self.goals[:, 1]


        x = init_pos
        N = len(x)
        if self.ordering_scheme == "fixed":
            self.init_ordering = torch.arange(0, self.num_nearest_goals, 1).repeat(N).view(N, self.num_nearest_goals)
        else:
            rel_to_goals = x.unsqueeze(1) - self.goals
            dists = torch.norm(rel_to_goals, dim = 2)
            _, smallest = torch.topk(dists, self.num_nearest_goals, dim=1, largest=False)
            self.init_ordering = smallest
        print(self.init_ordering)
        

        self.x = x
        self.step = 0

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
            scatter2 = ax.scatter(self.goal_xpoints, self.goal_ypoints, c='k', marker="x")

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

        ax = self.fig.add_subplot(1, num_col, 1)
        for arrow in self.arrows:
            ax.patches.remove(arrow) 

        self.arrows = []
        if action != None:
            _, max_per_agent = torch.max(action, dim = 1)
            #print(max_per_agent)
            print(set(max_per_agent.data.cpu().numpy()))
            
            for i in range(self.n_agents):
                x = self.x[i, 0]
                y = self.x[i, 1]
                goal = self.goals[ max_per_agent[i]]
                dx = goal[0] - x
                dy = goal[1] - y
                arrow = plt.Arrow(x, y, dx, dy )
                self.arrows.append(arrow)
                ax.add_patch(arrow)

        self.fig.canvas.draw()
        if not save_video:
            self.fig.canvas.flush_events()
            if action != None:
                plt.pause(0.01)
            else:
                plt.pause(0.01)

        return self.fig, self.scatter1

    def close(self):
        pass
