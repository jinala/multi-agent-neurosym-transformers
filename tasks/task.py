import sys

from tasks.random_task import *
from tasks.random_cross import *
from tasks.random_grid import *

def get_task_new(name, args):
	if name == "random_cross":
		return RandomCrossTask(args)
	elif name == "random_grid":
		return RandomGridTask(args)
	elif name == "unlabeled":
		return RandomTask(args)
	else:
		print("Unknown task name for task with args")
		assert(False)


def get_test_settings(name, num_groups=2):
	if name == "random_cross":
		# Each setting is list of groups that are present
		if num_groups == 2:
			return [[0], [1], [0,1]]
		elif num_groups == 4:
			return [[0], [1], [0,2], [0,1], [0,1,2], [0,1,2,3]]
			#[[0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2], [2,3], [1,3], [0,1,2], [0,1,3], [1,2,3], [0,2,3], [0,1,2,3]]
		elif num_groups == 6:
			return [[0], [2], [0,1], [0,2], [0,3,4], [0,2,3,5], [0,1,3,4,5], [0,1,2,3,4,5]]
		elif num_groups == 8:
			return [[0], [3], [0,1], [0,3], [0,3,4], [0,2,3,5], [0,1,3,4,5], [0,1,2,4,5,7], [0,1,2,3,4,5,6,7]]
		else:
			print("number of group not supported to get test settings")
			assert(False)
	elif name == "random_grid":
		return ['rlu', 'url', 'ull', 'rrd']
	elif name == "unlabeled":
		return [0]
	else:
		print("Unknown task name to get settings")
		assert(False)


def get_trainprog_settings(name, num_groups=2):
	if name == "random_cross":
		# Each setting is list of groups that are present
		if num_groups == 2:
			return [[0], [1], [0,1]]
		elif num_groups == 4:
			return [[0], [1], [0,2], [0,1], [0,1,2], [0,1,2,3]]
		elif num_groups == 6:
			return [[0], [2], [0,1], [0,2], [0,3,4], [0,2,3,5], [0,1,3,4,5], [0,1,2,3,4,5]]
		elif num_groups == 8:
			return [[0], [3], [0,1], [0,3], [0,3,4], [0,2,3,5], [0,1,3,4,5], [0,1,2,4,5,7], [0,1,2,3,4,5,6,7]]
		else:
			print("number of group not supported to get test settings")
			assert(False)
	elif name == "random_grid":
		return ['rlu', 'rld', 'url', 'drl', 'ull', 'rrd']
	elif name == "unlabeled":
		return [0]
	else:
		print("Unknown task name to get settings")
		assert(False)

def get_trainprog_settings_high(name, num_groups=2):
	if name == "random_cross":
		# Each setting is list of groups that are present
		if num_groups == 4:
			return [[0,1,3], [0,1,2], [1,2,3], [0,1,2,3]]
		else:
			print("number of group not supported to get test settings")
			assert(False)
	else:
		print("Unknown task name to get settings")
		assert(False)

def get_trainprog_settings_single(name, num_groups=2):
	if name == "random_cross":
		# Each setting is list of groups that are present
		if num_groups == 4:
			return [[0,1,2,3]]
		else:
			print("number of group not supported to get test settings")
			assert(False)
	else:
		print("Unknown task name to get settings")
		assert(False)

def get_trainprog_settings_balance(name, num_groups=2):
	if name == "random_cross":
		# Each setting is list of groups that are present
		if num_groups == 4:
			return [[2], [0,1], [0,1,2], [1,2,3], [0,1,2,3]]
		else:
			print("number of group not supported to get test settings")
			assert(False)
	else:
		print("Unknown task name to get settings")
		assert(False)
