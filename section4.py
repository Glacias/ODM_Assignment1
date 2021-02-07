import numpy as np

from section1 import *
from section2 import *
from section3 import *

class MDP_eq_estimate():
	def __init__(self, g, U, my_policy, f_transition, n_traj, size_traj, start_state):
		self.r_mat = np.zeros(g.shape + (len(U),))
		self.occur_mat = np.zeros(g.shape + (len(U),))
		self.transi_mat = np.zeros(g.shape + g.shape + (len(U),))

		# simulate n_traj, fill matrices
		for i in range(n_traj):
			self._simulate_traj(g, U, my_policy, f_transition, size_traj, start_state)

		# 0 occurence correspond to 0 values, set to 1 to avoid division problem
		self.occur_mat[self.occur_mat==0] = 1
		# divide reward and transition by occurences
		self.r_mat = np.true_divide(self.r_mat, self.occur_mat, where=(self.occur_mat != 0))
		self.transi_mat = np.true_divide(self.transi_mat, self.occur_mat, where=(self.occur_mat != 0))


	def _simulate_traj(self, g, U, my_policy, f_transition, size_traj, start_state):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action(x, my_policy, f_transition, g.shape)
			r = g[x_next]

			# get action index
			u = U.index(u)

			# fill matrices
			self.r_mat[x[0], x[1], u] += r
			self.occur_mat[x[0], x[1], u] += 1
			self.transi_mat[x_next[0], x_next[1], x[0], x[1], u] += 1

			x = x_next


	def _simulate_action(self, x, my_policy, f_transition, map_shape):
		u = my_policy.choose_action(x)
		x_next = f_transition(x, u, map_shape)

		return u, x_next

	# p(x_next | x, u)
	def p_transi(self, x_next, x, u):
		return self.transi_mat[x_next[0], x_next[1], x[0], x[1], u]

	# r(x, u)
	def r_state_action(self, x, u):
		return self.r_mat[x[0], x[1], u]

if __name__ == '__main__':

	# define problem's values
	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])
	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	gamma = 0.99

	# set values
	start_state = (3,0)
	random_policy = policy_rand(U)
	f_transition = (f_det, f_stoch)[1]
	n_traj = 10
	size_traj = 10000

	# estimate the MDP
	my_MDP_eq = MDP_eq_estimate(g, U, random_policy, f_transition, n_traj, size_traj, start_state)

	# test
	action = U.index((0,1))

	print(my_MDP_eq.p_transi((2,3), (2,2), action))
	print(my_MDP_eq.r_state_action((2,2), action))
