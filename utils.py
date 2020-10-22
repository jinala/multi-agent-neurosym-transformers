import numpy as np 
from scipy.optimize import linear_sum_assignment

def rand(range):
	a,b = range
	assert(a <= b)
	return (b-a)*np.random.random() + a

def norm(r):
	return r/(np.linalg.norm(r)  + 1e-4)

# match a point in S to a point in T such that sum of the distances between matched points is minimal 
# returns matching ind
def minimal_matching(S, T):
	assert(len(S) == len(T))
	cost_matrix = []
	for i in range(len(S)):
		row = []
		p1 = S[i]
		for j in range(len(T)):
			p2 = T[j]
			cost = np.linalg.norm(p1 - p2)
			row.append(cost)
		cost_matrix.append(row)

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	return col_ind

def box_formation(num_robots, start = np.array([0.0, 0.0]), end = np.array([10.0, 0.0]), width = 1.0, diff = 0.5, need_counter=False):
	a = norm(end - start)
	b = np.array([a[1], -a[0]])

	l = np.linalg.norm(end - start)

	goals = []
	counter = 0
	while len(goals) < num_robots and counter < num_robots*100:
		x = rand((0, l))
		y = rand((-width, width))
		point = start + a*x + b*y
		if not check_collisions(point, goals, diff):
			goals.append(point)
		counter += 1

	while len(goals) < num_robots:
		x = rand((0, l))
		y = rand((-width, width))
		point = start + a*x + b*y
		goals.append(point)

	if need_counter:
		return goals, counter
	else:
		return goals


def check_collisions(r, points, diff):
	for p in points:
		dist = np.linalg.norm(r - p)
		if dist < diff:
			return True
	return False
