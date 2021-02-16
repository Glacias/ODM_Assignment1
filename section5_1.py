import numpy as np
import matplotlib.pyplot as plt

from section1 import *
from section2 import *
from section3 import *

# Class for generating the policy based on Q-learning
class Offline_Q_learning():
	def __init__(self, g, U, my_policy, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, J_opt):
		# Action set
		self.U = U
		# Infinite norm of the difference between estimated Q and J_mu optimal for each transition
		self.gap_to_opti = [None]*size_traj

		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, my_policy, f_transition, size_traj, start_state, learning_rate, gamma, J_opt)


	def _simulate_traj(self, g, my_policy, f_transition, size_traj, start_state, learning_rate, gamma, J_opt):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action(x, my_policy, f_transition, g.shape)
			r = g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# Update Q
			self.Q[x[0], x[1], u_idx] = (1-learning_rate)*self.Q[x[0], x[1], u_idx] + learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			# Compute infinite norm
			self.gap_to_opti[i] = np.abs(self.Q.max(axis=2) - J_opt).max()

			x = x_next

	def _simulate_action(self, x, my_policy, f_transition, map_shape):
		u = my_policy.choose_action(x)
		x_next = f_transition(x, u, map_shape)

		return u, x_next


if __name__ == '__main__':
	# choose case : 0 for det and 1 for stoch
	case = 1

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
	expected_return = [expected_ret_det, expected_ret_stoch][case]
	f_transition = (f_det, f_stoch)[case]
	n_traj = 1
	size_traj = 10000000
	learning_rate = 0.05

	## Get the J for the optimal policy
	# MDP
	if case == 0:
		original_MDP = MDP_eq_det(g)
	else:
		original_MDP = MDP_eq_stoch(g)

	# compute N expected large enough
	Br = np.abs(g).max()
	thresh = 0.1
	N_J = compute_N(gamma, Br, thresh)
	thresh = 1
	print("Chosen N : " + str(N_J))
	print()

	# Q_N for the original MDP
	Q_orig, _ = compute_Q_dyna(g, U, gamma, N_J, original_MDP)
	# derive a policy from the Q_N matrix of the original MDP
	policy_mat_orig = get_optimal_pol_mat(Q_orig)

	# set policies
	policy_Q_orig = policy_set(U, policy_mat_orig)

	# compute the expected returns (J) for the optimal policy
	J_opt = compute_J_dyna(g, U, policy_Q_orig, gamma, N_J, expected_return)
	print("J_N of the original MDP (N = {}) :".format(N_J))
	print(J_opt)


	## Offline Q-learning policy
	off_Q_learn = Offline_Q_learning(g, U, my_policy, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, J_opt)

	# Q approximate
	Q = off_Q_learn.Q
	print("Offline estimation Q function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("Offline Q-learning policy :")
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
	expected_return = [expected_ret_det, expected_ret_stoch][case]

	# compute N expected large enough
	Br = g.max()
	thresh = 0.1
	max_N = N_J
	print("Chosen N : " + str(max_N))
	print()

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, max_N, expected_return)
	print("J_N of the offline policy :")
	print(J_opt)

	# Graph of the infinite norm
	plt.plot(range(0,size_traj), off_Q_learn.gap_to_opti)
	if case == 0:
		plt.title('Offline Q-learning (deterministic)')
	else:
		plt.title('Offline Q-learning (stochastic)')
	plt.xlabel('T (Number of transition)')
	plt.ylabel('$\|\| \hat{Q} - J^{\mu^*_N}_N \|\|_\infty$')
	plt.show()