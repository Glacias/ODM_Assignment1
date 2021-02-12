import numpy as np

from section1 import *
from section2 import *
from section3 import *

class MDP_eq_estimate():
	def __init__(self, g, U, my_policy, f_transition, n_traj, size_traj, start_state):
		self.U = U
		self.r_mat = np.zeros(g.shape + (len(U),))
		self.occur_mat = np.zeros(g.shape + (len(U),))
		self.transi_mat = np.zeros(g.shape + g.shape + (len(U),))

		# simulate n_traj, fill matrices
		for i in range(n_traj):
			self._simulate_traj(g, my_policy, f_transition, size_traj, start_state)

		"""
		# 0 occurence correspond to 0 values, set to 1 to avoid division problem
		self.occur_mat[self.occur_mat==0] = 1
		# divide reward and transition by occurences
		self.r_mat = np.true_divide(self.r_mat, self.occur_mat, where=(self.occur_mat != 0))
		self.transi_mat = np.true_divide(self.transi_mat, self.occur_mat, where=(self.occur_mat != 0))
		"""


	def _simulate_traj(self, g, my_policy, f_transition, size_traj, start_state):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action(x, my_policy, f_transition, g.shape)
			r = g[x_next]

			# get action index
			u = self.U.index(u)

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
		# get action index (from action tuple)
		u = self.U.index(u)

		# if <state, action> pair never occured -> uniform probability
		if self.occur_mat[x[0], x[1], u] == 0:
			return 1 / (self.occur_mat.shape[0] * self.occur_mat.shape[1])
		# else get prob from matrices
		else:

			return self.transi_mat[x_next[0], x_next[1], x[0], x[1], u] / self.occur_mat[x[0], x[1], u]

	# r(x, u)
	def r_state_action(self, x, u):
		# get action index (from action tuple)
		u = self.U.index(u)

		# if <state, action> pair never occured -> ??? (0 average reward for now)
		if self.occur_mat[x[0], x[1], u] == 0:
			return 0

		# else get average reward from matrices
		else :
			return self.r_mat[x[0], x[1], u] / self.occur_mat[x[0], x[1], u]

	## NOT TESTED
	def expected_ret_est(self, g, U, x, my_policy, gamma, J_prev):
		u = my_policy.choose_action(x)

		J_x_tot = self.r_state_action(x, u)

		# average over all possible next state (ponderate with respective prob)
		for k in range(J_prev.shape[0]):
			for l in range(J_prev.shape[1]):
				J_x_tot += self.p_transi((k,l), x, u) * gamma * J_prev[(k,l)]


		return J_x_tot

if __name__ == '__main__':
	# choose case : 0 for det and 1 for stoch
	case = 0

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
	my_policy = policy_rand(U)
	#my_policy = policy_cst(U, "right")
	f_transition = (f_det, f_stoch)[case]
	n_traj = 10
	size_traj = 10000

	# estimate the MDP
	my_MDP_eq = MDP_eq_estimate(g, U, my_policy, f_transition, n_traj, size_traj, start_state)

	# test
	action = (0,1)

	print(my_MDP_eq.p_transi((2,3), (2,2), action))
	print(my_MDP_eq.r_state_action((2,2), action))

	# compute N expected large enough
	Br = g.max()
	thresh = 0.1
	max_N = compute_N_bis(gamma, Br, thresh)
	print("Chosen N : " + str(max_N))
	print()

	# Q approximate
	Q, _ = compute_Q_dyna(g, U, gamma, max_N, my_MDP_eq)
	print("Estimation Q_N function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("policy :")
	policy_mat = get_optimal_pol_mat(Q)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat.shape[0]):
		for l in range(policy_mat.shape[1]):
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)

	#Recompute new with first rule for J ?
	# --- --- --- --- ---

	# compute the expected returns (J)
	J = compute_J_dyna(g, U, policy_Q, gamma, max_N, my_MDP_eq.expected_ret_est)

	print(J)