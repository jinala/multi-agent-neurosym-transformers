import sys 
import numpy as np 
from utils import *
import matplotlib.cm as cm


class RandomTask():
	def __init__(self, args):
		pass 

	def sample_task(self, args):
		self.num_robots = args['num_agents']
		n = self.num_robots
		half = 2.0

		init_pos = box_formation(n, start = np.array([-half, 0]), end = np.array([half, 0]), width = half, diff = 1.0)
		half = half * 4.0
		
		self.goals = box_formation(n, start = np.array([-half, 0]), end = np.array([half, 0]), width = half, diff = 2.0)

		return init_pos, self.goals, []


	def gen_robots(self):
		robots = []
		robot_args = []
		n = self.num_robots

		init_pos, goals = self.sample_task(n)
		
		for i in range(n):
			init_state = np.array([init_pos[i][0], init_pos[i][1], 0.0, 0.0])
			r = Robot(init_state)
			robots.append(r)

			args = [goals[i]]
			robot_args.append(args)
		return robots, robot_args

	def get_plot_lims(self):
		return [-1, self.num_robots*0.5 + 1.0, - 1, self.num_robots*0.5 + 1]

	# plot env excluding the robots	
	def plot(self):
		return 

	def robot_colors(self):
		return cm.rainbow(np.linspace(0, 1, self.num_robots))

