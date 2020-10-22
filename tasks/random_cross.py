import sys
import numpy as np
from utils import *
import random
import matplotlib.cm as cm
import matplotlib.pyplot as pl 

SEQUENTIAL_COLORS = [
             'Blues',   'Reds', 'Greens', 'Purples', 'Oranges', 'Greys',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


class RandomCrossTask(object):
	"""docstring for RandomCrossTask."""

	def __init__(self, args):
		super(RandomCrossTask, self).__init__()
		self.box_width_ratio = args['box_width_ratio']
		self.l = args['l']
		num_groups = args['num_groups']
		assert num_groups == 2 or num_groups == 4 or num_groups == 6 or num_groups == 8, 'RandomCrossTask only supports number of groups 2,4,6,8'
		self.num_groups = num_groups


	def sample_task(self, args):
		'''
		prob_group: probability that each group is present (after the first one)
		'''
		# Unpack args
		seed = args['seed'] if 'seed' in args else -1
		fixed_setting = args['setting'] if 'setting' in args else None
		n_robots_per_group = args['num_agents']
		prob_group = args['prob_group'] if 'prob_group' in args else 0.33
		

		# Get box positions
		self.n_robots_per_group = n_robots_per_group
		self.angs = np.arange(0, 2.0*np.pi, 2.0*np.pi/self.num_groups).tolist()

		if fixed_setting:
			self.group_ids = fixed_setting

		else:
			# Sample groups
			self.group_ids = []
		
			first_group = np.random.randint(self.num_groups)
			self.group_ids.append(first_group)
			for i in range(self.num_groups):
				if not i == first_group:
					if np.random.rand() < prob_group:
						self.group_ids.append(i)

		if seed > 0:
			np.random.seed(seed)

		# Sample locations within groups
		init_pos = []
		goal_pos = []
		width = n_robots_per_group/self.box_width_ratio
		l = self.l
		for i in self.group_ids:
			ia = self.angs[i]
			init_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width]), end = np.array([-l, width]), width = width)
			ca = np.cos(ia)
			sa = np.sin(ia)
			R = np.array([[ca, sa], [-sa, ca]])
			init_pos_1 = [np.dot(x, R) for x in init_pos_1]
			init_pos.extend(init_pos_1)

			
			ga = self.angs[i] + np.pi
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width]), end = np.array([-l, width]), width = width)
			ca = np.cos(ga)
			sa = np.sin(ga)
			R = np.array([[ca, sa], [-sa, ca]])
			goal_pos_1 = [np.dot(x, R) for x in goal_pos_1]
			goal_pos.extend(goal_pos_1)

		return init_pos, goal_pos, lambda x : 0

	def plot(self):
		return

	def get_plot_lims(self):
		lim = self.l + self.n_robots_per_group / self.box_width_ratio + 2.0
		return [-lim, lim, -lim, lim]

	def robot_colors(self):
		n = self.n_robots_per_group
		n_groups = len(self.group_ids)
		colors = []
		for i in range(n_groups):
			colors.extend(pl.get_cmap(SEQUENTIAL_COLORS[i])(np.linspace(0.8, 0.4, n)))
		return colors
