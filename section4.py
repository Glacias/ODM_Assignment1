import numpy as np
import matplotlib.pyplot as plt

from section1 import *
from section2 import *
from section3 import *

class MDP_eq_estimate(MDP_eq):
	def __init__(self, g, U, my_policy, f_transition, n_traj, size_traj, start_state):
		self.g = g
		self.U = U
		self.r_mat = np.zeros(g.shape + (len(U),))
		self.occur_mat = np.zeros(g.shape + (len(U),))
		self.transi_mat = np.zeros(g.shape + g.shape + (len(U),))

		# simulate n_traj, fill matrices
		for i in range(n_traj):
			self.simulate_traj(my_policy, f_transition, size_traj, start_state)

		"""
		# 0 occurence correspond to 0 values, set to 1 to avoid division problem
		self.occur_mat[self.occur_mat==0] = 1
		# divide reward and transition by occurences
		self.r_mat = np.true_divide(self.r_mat, self.occur_mat, where=(self.occur_mat != 0))
		self.transi_mat = np.true_divide(self.transi_mat, self.occur_mat, where=(self.occur_mat != 0))
		"""


	def simulate_traj(self, my_policy, f_transition, size_traj, start_state):
		x = start_state

		# default value if empty traj
		x_next = x

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action(x, my_policy, f_transition, self.g.shape)
			r = self.g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# fill matrices
			self.r_mat[x[0], x[1], u_idx] += r
			self.occur_mat[x[0], x[1], u_idx] += 1
			self.transi_mat[x_next[0], x_next[1], x[0], x[1], u_idx] += 1

			x = x_next

		return x_next


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
		u_idx = self.U.index(u)

		# if <state, action> pair never occured -> ??? (0 average reward for now)
		if self.occur_mat[x[0], x[1], u_idx] == 0:
			return 0

		# else get average reward from matrices
		else :
			return self.r_mat[x[0], x[1], u_idx] / self.occur_mat[x[0], x[1], u_idx]



def compare_p(U, map_shape, MDP_original, MDP_est):
	# get the infinite norm (max absolute)
	# of the diff between the original and the estimate
	max_val = -float("inf")
	for k_next in range(map_shape[0]):
		for l_next in range(map_shape[1]):
			for k in range(map_shape[0]):
				for l in range(map_shape[1]):
					for u in U:
						curr_diff = abs(MDP_original.p_transi((k_next, l_next), (k, l), u) - MDP_est.p_transi((k_next, l_next), (k, l), u))
						if curr_diff > max_val:
							max_val = curr_diff

	return max_val

def compare_r(U, map_shape, MDP_original, MDP_est):
	# get the infinite norm (max absolute)
	# of the diff between the original and the estimate
	max_val = -float("inf")
	for k in range(map_shape[0]):
		for l in range(map_shape[1]):
			for u in U:
				curr_diff = abs(MDP_original.r_state_action((k, l), u) - MDP_est.r_state_action((k, l), u))
				if curr_diff > max_val:
					max_val = curr_diff

	return max_val

def compare_Q(g, U, gamma, N, map_shape, MDP_original, MDP_est):
	# get the infinite norm (max absolute)
	# of the diff between the original and the estimate
	Q_orig, _ = compute_Q_dyna(g, U, gamma, N, MDP_original)
	Q_est, _ = compute_Q_dyna(g, U, gamma, N, MDP_est)


	max_val = -float("inf")
	for k in range(map_shape[0]):
		for l in range(map_shape[1]):
			for u_idx in range(len(U)):
				curr_diff = abs(Q_orig[k, l, u_idx] - Q_est[k, l, u_idx])
				if curr_diff > max_val:
					max_val = curr_diff

	return max_val



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
	n_traj = 1
	#size_traj = 10000

	# compute N expected large enough for Q
	Br = np.abs(g).max()
	thresh = 1
	N_Q = compute_N_bis(gamma, Br, thresh)

	### CONVERGENCE SPEED (+ infinite norm Q)
	max_power = 6
	if case == 0:
		original_MDP = MDP_eq_det(g)
	else:
		original_MDP = MDP_eq_stoch(g)

	## traj size 1
	print("Compute for a traj of size 1 :")
	init_size = 1
	traj_sizes = [1]
	# estimate the MDP
	print("Simulation  .")
	est_MDP = MDP_eq_estimate(g, U, my_policy, f_transition, n_traj, init_size, start_state)

	# infinite norm
	p_conv = []
	r_conv = []
	Q_conv = []
	print("Comparing p ..")
	p_conv.append(compare_p(U, g.shape, original_MDP, est_MDP))
	print("Comparing r ...")
	r_conv.append(compare_r(U, g.shape, original_MDP, est_MDP))
	print("Comparing q ....")
	Q_conv.append(compare_Q(g, U, gamma, N_Q, g.shape, original_MDP, est_MDP))


	# size from 10 to 1M
	for traj_size_power in range(1, max_power+1):
		print("Compute for a traj of size {} :".format(int(10**traj_size_power)))
		print("Simulation  .")
		added_steps = (10**traj_size_power) - (10**(traj_size_power-1))
		start_state = est_MDP.simulate_traj(my_policy, f_transition, added_steps, start_state)

		# append convergence
		traj_sizes.append(10**traj_size_power)
		print("Comparing p ..")
		p_conv.append(compare_p(U, g.shape, original_MDP, est_MDP))
		print("Comparing r ...")
		r_conv.append(compare_r(U, g.shape, original_MDP, est_MDP))
		print("Comparing q ....")
		Q_conv.append(compare_Q(g, U, gamma, N_Q, g.shape, original_MDP, est_MDP))

	fig, axs = plt.subplots(2, sharex=True)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.suptitle(r'$L_\infty$')
	plt.xscale("log")
	axs[0].plot(traj_sizes, p_conv)
	axs[0].set_ylim(bottom=0)
	axs[0].set(ylabel=r'$\hat{p} - p$')
	axs[1].plot(traj_sizes, r_conv)
	axs[1].set_ylim(bottom=0)
	axs[1].set(xlabel='trajectory length' ,ylabel=r'$\hat{r} - r$')
	plt.show()

	print()
	print("length considered :")
	print(traj_sizes)
	print("Infinity norm of the difference between Q and Q hat :")
	print(Q_conv)

	print()
	### OPTIMAL POLICY FOR THE ESTIMATE

	# Q_N for the estimate MDP
	Q_est, _ = compute_Q_dyna(g, U, gamma, N_Q, est_MDP)
	print("Estimation Q_N(u, x) function (N = {}) for the estimate MDP :".format(N_Q))
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q_est, 2, 0))
	print()

	# derive a policy from the Q_N matrix of the estimate MDP
	print("Optimal policy for the estimate MDP:")
	policy_mat_est = get_optimal_pol_mat(Q_est)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat_est.shape[0]):
		for l in range(policy_mat_est.shape[1]):
			print(instruction_arrow[policy_mat_est[k,l]], end="")
		print()


	print()
	### J FOR BOTH OPTIMAL POLICIES (ORIG/EST)
	expected_return_orig = [expected_ret_det, expected_ret_stoch][case]
	thresh = 0.1
	N_J = compute_N(gamma, Br, thresh)

	# Q_N for the original MDP
	Q_orig, _ = compute_Q_dyna(g, U, gamma, N_Q, original_MDP)
	# derive a policy from the Q_N matrix of the original MDP
	policy_mat_orig = get_optimal_pol_mat(Q_orig)

	# set policies
	policy_Q_orig = policy_set(U, policy_mat_orig)
	policy_Q_est = policy_set(U, policy_mat_est)

	# compute the expected returns (J)
	J_opt_orig = compute_J_dyna(g, U, policy_Q_orig, gamma, N_J, expected_return_orig)
	print("J_N of the original MDP (N = {}) :".format(N_J))
	print(J_opt_orig)
	# NB which expected return to use for mu^? (original or estimated)
	J_opt_est = compute_J_dyna(g, U, policy_Q_est, gamma, N_J, expected_return_orig)
	print("J_N of the estimate MDP (N = {}) :".format(N_J))
	print(J_opt_est)
