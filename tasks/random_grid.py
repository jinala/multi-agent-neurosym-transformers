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

LEFT = 0
RIGHT = 1
UP = 3
DOWN = 4
MID = 5

class RandomGridTask(object):
	"""docstring for RandomGridTask."""

	def __init__(self, args):
		super(RandomGridTask, self).__init__()
		self.box_width_ratio = args['box_width_ratio']
		self.l = args['l']
		self.num_groups = 3

	def sample_task(self, args):
		'''
		prob_go_mid: probability that side groups go to mid
		'''
		# Unpack args
		seed = args['seed'] if 'seed' in args else -1
		fixed_setting = args['setting'] if 'setting' in args else None
		n_robots_per_group = args['num_agents']
		prob_go_mid = args['prob_go_mid'] if 'prob_go_mid' in args else 0.5

		self.n_robots_per_group = n_robots_per_group

		if fixed_setting:
			left_direction, mid_direction, right_direction = self.direction_from_spec(fixed_setting)
		else:
			# Sample where to go
			mid_direction = np.random.randint(2)
			left_direction = self.sample_side_direction(prob_go_mid)
			right_direction = self.sample_side_direction(prob_go_mid)
			if left_direction == MID and right_direction == MID:
				if np.random.rand() < 0.5:
					left_direction = self.sample_side_direction(0.0)
				else:
					right_direction = self.sample_side_direction(0.0)

		assert left_direction != MID or right_direction != MID, "Conflict of directions"

		if seed > 0:
			np.random.seed(seed)

		# Sample locations within groups
		init_pos = []
		goal_pos = []
		width = n_robots_per_group/self.box_width_ratio
		l = self.l

		# Left
		init_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width]), end = np.array([-l, width]), width = width)
		init_pos.extend(init_pos_1)
		if left_direction == UP:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width+l]), end = np.array([-l, width+l]), width = width)
			goal_pos.extend(goal_pos_1)
		elif left_direction == DOWN:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width-l]), end = np.array([-l, width-l]), width = width)
			goal_pos.extend(goal_pos_1)
		else:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([0, -width]), end = np.array([0, width]), width = width)
			goal_pos.extend(goal_pos_1)

		# Right
		init_pos_1 = box_formation(n_robots_per_group, start = np.array([l, -width]), end = np.array([l, width]), width = width)
		init_pos.extend(init_pos_1)
		if right_direction == UP:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([l, -width+l]), end = np.array([l, width+l]), width = width)
			goal_pos.extend(goal_pos_1)
		elif right_direction == DOWN:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([l, -width-l]), end = np.array([l, width-l]), width = width)
			goal_pos.extend(goal_pos_1)
		else:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([0, -width]), end = np.array([0, width]), width = width)
			goal_pos.extend(goal_pos_1)

		# Mid
		init_pos_1 = box_formation(n_robots_per_group, start = np.array([0, -width]), end = np.array([0, width]), width = width)
		init_pos.extend(init_pos_1)
		if mid_direction == LEFT:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([-l, -width]), end = np.array([-l, width]), width = width)
			goal_pos.extend(goal_pos_1)
		else:
			goal_pos_1 = box_formation(n_robots_per_group, start = np.array([l, -width]), end = np.array([l, width]), width = width)
			goal_pos.extend(goal_pos_1)

		return init_pos, goal_pos, lambda x : 0

	def plot(self):
		return

	def get_plot_lims(self):
		lim = self.l + self.n_robots_per_group / self.box_width_ratio + 2.0
		return [-lim, lim, -lim, lim]

	def robot_colors(self):
		n = self.n_robots_per_group
		n_groups = self.num_groups
		colors = []
		for i in range(n_groups):
			colors.extend(pl.get_cmap(SEQUENTIAL_COLORS[i])(np.linspace(0.8, 0.4, n)))
		return colors

	def sample_side_direction(self, prob_go_mid):
		if np.random.rand() < prob_go_mid:
			direction = MID
		else:
			if np.random.rand() < 0.5:
				direction = UP
			else:
				direction = DOWN

		return direction

	def direction_from_spec(self, spec):
		left_spec = spec[0]
		if left_spec == 'u':
			left_direction = UP
		elif left_spec == 'd':
			left_direction = DOWN
		elif left_spec == 'r':
			left_direction = MID
		else:
			raise ValueError("Left direction spec wrong")

		mid_spec = spec[1]
		if mid_spec == 'l':
			mid_direction = LEFT
		elif mid_spec == 'r':
			mid_direction = RIGHT
		else:
			raise ValueError("Mid direction spec wrong")

		right_spec = spec[2]
		if right_spec == 'u':
			right_direction = UP
		elif right_spec == 'd':
			right_direction = DOWN
		elif right_spec == 'l':
			right_direction = MID
		else:
			raise ValueError("Right direction spec wrong")

		return left_direction, mid_direction, right_direction
