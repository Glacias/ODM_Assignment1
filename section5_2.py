import numpy as np
import math
import matplotlib.pyplot as plt

from section1 import *
from section2 import *
from section3 import *

## Class for generating the policy based on Q-learning
# First experimental protocol
class Online_Q_learning_v1():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt):
		# Action set
		self.U = U
		# Infinite norm of the difference between estimated Q and J_mu optimal for each episode
		self.gap_to_opti = [None]*n_traj
		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			# Simulate traj
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g)
			# Compute infinite norm
			self.gap_to_opti[i] = np.abs(self.Q.max(axis=2) - J_opt).max()

	def _simulate_traj(self, g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action_greedy(x, f_transition, g.shape, epsilon_g)
			r = g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# Update Q
			self.Q[x[0], x[1], u_idx] = (1-learning_rate)*self.Q[x[0], x[1], u_idx] + learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			x = x_next

	def _simulate_action_greedy(self, x, f_transition, map_shape, epsilon_g):
		if np.random.rand() < epsilon_g:
			# Exploration
			u_idx = np.random.randint(0,len(self.U))
		else:
			# Exploitation
			u_idx = np.argmax(self.Q[x[0],x[1]])

		u = self.U[u_idx]
		x_next = f_transition(x, u, map_shape)

		return u, x_next



# Second experimental protocol
class Online_Q_learning_v2():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt):
		# Action set
		self.U = U
		# Learning rate
		self.learning_rate = learning_rate
		# Infinite norm of the difference between estimated Q and J_mu optimal for each episode
		self.gap_to_opti = [None]*n_traj
		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, gamma, epsilon_g)
			# Compute infinite norm
			self.gap_to_opti[i] = np.abs(self.Q.max(axis=2) - J_opt).max()

	def _simulate_traj(self, g, f_transition, size_traj, start_state, gamma, epsilon_g):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action_greedy(x, f_transition, g.shape, epsilon_g)
			r = g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# Update Q
			self.Q[x[0], x[1], u_idx] = (1-self.learning_rate)*self.Q[x[0], x[1], u_idx] + self.learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			# Decay the learning rate
			self.learning_rate = 0.8*self.learning_rate

			x = x_next

	def _simulate_action_greedy(self, x, f_transition, map_shape, epsilon_g):
		if np.random.rand() < epsilon_g:
			# Exploration
			u_idx = np.random.randint(0,len(self.U))
		else:
			# Exploitation
			u_idx = np.argmax(self.Q[x[0],x[1]])

		u = self.U[u_idx]
		x_next = f_transition(x, u, map_shape)

		return u, x_next



# Third experimental protocol
class Online_Q_learning_v3():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt):
		# Action set
		self.U = U
		# Initialize buffer
		self.buffer = []
		# Infinite norm of the difference between estimated Q and J_mu optimal for each episode
		self.gap_to_opti = [None]*n_traj
		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g)
			# Compute infinite norm
			self.gap_to_opti[i] = np.abs(self.Q.max(axis=2) - J_opt).max()

	def _simulate_traj(self, g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action_greedy(x, f_transition, g.shape, epsilon_g)
			r = g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# Update Q
			self.Q[x[0], x[1], u_idx] = (1-learning_rate)*self.Q[x[0], x[1], u_idx] + learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			# Create a tuple and add it to the buffer
			self.buffer.append((x, u_idx, r, x_next))

			# Length of the buffer
			length_buff = len(self.buffer)

			# Make 10 draw from the buffer and update Q
			for k in range(10):
				# Draw a sample
				t_idx = np.random.randint(0, length_buff)

				# Extract the tuple
				x = self.buffer[t_idx][0]
				u_idx = self.buffer[t_idx][1]
				r = self.buffer[t_idx][2]
				x_next = self.buffer[t_idx][3]

				# Update Q
				self.Q[x[0], x[1], u_idx] = (1-learning_rate)*self.Q[x[0], x[1], u_idx] + learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			# Move to next transition
			x = x_next

	def _simulate_action_greedy(self, x, f_transition, map_shape, epsilon_g):
		if np.random.rand() < epsilon_g:
			# Exploration
			u_idx = np.random.randint(0,len(self.U))
		else:
			# Exploitation
			u_idx = np.argmax(self.Q[x[0],x[1]])

		u = self.U[u_idx]
		x_next = f_transition(x, u, map_shape)

		return u, x_next


# Q-learning with another exploration policy
# New polocy : decaying e-greddy value
class Online_Q_learning_v4():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt):
		# Action set
		self.U = U
		# Epsilon_g
		self.epsilon_g = epsilon_g
		# Infinite norm of the difference between estimated Q and J_mu optimal for each episode
		self.gap_to_opti = [None]*n_traj
		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma)
			# Compute infinite norm
			self.gap_to_opti[i] = np.abs(self.Q.max(axis=2) - J_opt).max()
			# Reset epsilon_g
			self.epsilon_g = epsilon_g

	def _simulate_traj(self, g, f_transition, size_traj, start_state, learning_rate, gamma):
		x = start_state

		# simulate a traj
		for i in range(size_traj):
			# simulate an action
			u, x_next = self._simulate_action_greedy(x, f_transition, g.shape)
			r = g[x_next]

			# get action index
			u_idx = self.U.index(u)

			# Update Q
			self.Q[x[0], x[1], u_idx] = (1-learning_rate)*self.Q[x[0], x[1], u_idx] + learning_rate*(r + gamma*self.Q[x_next[0], x_next[1], :].max())

			x = x_next

	def _simulate_action_greedy(self, x, f_transition, map_shape):
		if np.random.rand() < self.epsilon_g:
			# Exploration
			u_idx = np.random.randint(0,len(self.U))
		else:
			# Exploitation
			u_idx = np.argmax(self.Q[x[0],x[1]])

		u = self.U[u_idx]
		x_next = f_transition(x, u, map_shape)

		# Decaying epsilon_g
		self.epsilon_g = 0.9*self.epsilon_g
		#print(self.epsilon_g)

		return u, x_next


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
	gamma = 0.4

	# set values
	start_state = (3,0)
	my_policy = policy_rand(U)
	#my_policy = policy_cst(U, "right")
	f_transition = (f_det, f_stoch)[case]
	expected_return = [expected_ret_det, expected_ret_stoch][case]
	n_traj = 100
	size_traj = 1000
	learning_rate = 0.05
	epsilon_g = 0.25

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
	N_Q = compute_N_bis(gamma, Br, thresh)
	print("Chosen N : " + str(N_J))
	print()

	# Q_N for the original MDP
	Q_orig, _ = compute_Q_dyna(g, U, gamma, N_Q, original_MDP)
	# derive a policy from the Q_N matrix of the original MDP
	policy_mat_orig = get_optimal_pol_mat(Q_orig)

	# set policies
	policy_Q_orig = policy_set(U, policy_mat_orig)

	# compute the expected returns (J) for the optimal policy
	J_opt = compute_J_dyna(g, U, policy_Q_orig, gamma, N_J, expected_return)
	print("J_N of the original MDP (N = {}) :".format(N_J))
	print(J_opt)


	## First experimental protocol
	print("\n--First experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v1 = Online_Q_learning_v1(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt)

	# Q approximate
	Q_v1 = on_Q_learn_v1.Q
	print("Online estimation Q function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q_v1, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("Online Q-learning policy :")
	policy_mat = get_optimal_pol_mat(Q_v1)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat.shape[0]):
		for l in range(policy_mat.shape[1]):
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, N_J, expected_return)
	print("J_N of the online policy :")
	print(J_opt)

	# Graph of the infinite norm
	plt.plot(range(0,n_traj), on_Q_learn_v1.gap_to_opti)
	if case == 0:
		plt.title('First protocol (deterministic)')
	else:
		plt.title('First protocol (stochastic)')
	plt.xlabel('Number of episode')
	plt.ylabel('$\|\| \hat{Q} - J^{\mu^*_N}_N \|\|_\infty$')
	plt.show()


	## Second experimental protocol
	print("\n--Second experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v2 = Online_Q_learning_v2(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt)

	# Q approximate
	Q_v2 = on_Q_learn_v2.Q
	print("Online estimation Q function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q_v2, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("Online Q-learning policy :")
	policy_mat = get_optimal_pol_mat(Q_v2)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat.shape[0]):
		for l in range(policy_mat.shape[1]):
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, N_J, expected_return)
	print("J_N of the online policy :")
	print(J_opt)

	# Graph of the infinite norm
	plt.plot(range(0,n_traj), on_Q_learn_v2.gap_to_opti)
	if case == 0:
		plt.title('Second protocol (deterministic)')
	else:
		plt.title('Second protocol (stochastic)')
	plt.xlabel('Number of episode')
	plt.ylabel('Infinite norm')
	plt.show()


	## Third experimental protocol
	print("\n--Third experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v3 = Online_Q_learning_v3(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g, J_opt)

	# Q approximate
	Q_v3 = on_Q_learn_v3.Q
	print("Online estimation Q function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q_v3, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("Online Q-learning policy :")
	policy_mat = get_optimal_pol_mat(Q_v3)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat.shape[0]):
		for l in range(policy_mat.shape[1]):
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, N_J, expected_return)
	print("J_N of the online policy :")
	print(J_opt)

	# Graph of the infinite norm
	plt.plot(range(0,n_traj), on_Q_learn_v3.gap_to_opti)
	if case == 0:
		plt.title('Third protocol (deterministic)')
	else:
		plt.title('Third protocol (stochastic)')
	plt.xlabel('Number of episode')
	plt.ylabel('$\|\| \hat{Q} - J^{\mu^*_N}_N \|\|_\infty$')
	plt.show()



	## Fourth experimental protocol (bonus)
	print("\n--Fourth experimental protocol (bonus)--\n")

	# Initial epsilon_g
	epsilon_g = 1
	learning_rate = 0.05

	# Online Q-learning policy
	on_Q_learn_v4 = Online_Q_learning_v4(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, 1, J_opt)

	# Q approximate
	Q_v4 = on_Q_learn_v4.Q
	print("Online estimation Q function (u, x) :")
	# move axis such that Q is displayed by action u on the first axis
	print(np.moveaxis(Q_v4, 2, 0))
	print()

	# derive a policy from the Q_N matrix
	print("Online Q-learning policy :")
	policy_mat = get_optimal_pol_mat(Q_v4)

	# display policy with arrow
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(policy_mat.shape[0]):
		for l in range(policy_mat.shape[1]):
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, N_J, expected_return)
	print("J_N of the online policy :")
	print(J_opt)

	# Graph of the infinite norm
	plt.plot(range(0,n_traj), on_Q_learn_v4.gap_to_opti)
	if case == 0:
		plt.title('Fourth protocol (deterministic)')
	else:
		plt.title('Fourth protocol (stochastic)')
	plt.xlabel('Number of episode')
	plt.ylabel('Infinite norm')
	plt.show()