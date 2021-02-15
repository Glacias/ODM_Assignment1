import numpy as np
import math

from section1 import *
from section2 import *
from section3 import *

## Class for generating the policy based on Q-learning
# First experimental protocol
class Online_Q_learning_v1():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g):
		# Action set
		self.U = U

		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g)


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
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g):
		# Action set
		self.U = U
		# Learning rate
		self.learning_rate = learning_rate

		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, gamma, epsilon_g)


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
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g):
		# Action set
		self.U = U

		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma, epsilon_g)

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


# Q-learning with another exploration policy
# New polocy : decaying e-greddy value
class Online_Q_learning_v4():
	def __init__(self, g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g):
		# Action set
		self.U = U
		# Epsilon_g
		self.epsilon_g = epsilon_g

		# Initialize the Q matrix to 0 everywhere
		self.Q = np.zeros(g.shape + (len(U),))

		# Simulate n_traj and fill Q
		for i in range(n_traj):
			self._simulate_traj(g, f_transition, size_traj, start_state, learning_rate, gamma)
			# Reset epsilon_g
			#self.epsilon_g = epsilon_g

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
		self.epsilon_g = 0.9999*self.epsilon_g
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
	gamma = 0.99

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

	# compute N expected large enough
	Br = g.max()
	thresh = 0.1
	max_N = compute_N_bis(gamma, Br, thresh)
	print("Chosen N : " + str(max_N))
	print()



	## First experimental protocol
	print("\n--First experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v1 = Online_Q_learning_v1(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g)

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
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, max_N, expected_return)
	print("J_N of the online policy :")
	print(J_opt)



	## Second experimental protocol
	print("\n--Second experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v2 = Online_Q_learning_v2(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g)

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
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, max_N, expected_return)
	print("J_N of the online policy :")
	print(J_opt)



	## Third experimental protocol
	print("\n--Third experimental protocol--\n")

	# Online Q-learning policy
	on_Q_learn_v2 = Online_Q_learning_v2(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, epsilon_g)

	# Q approximate
	Q_v3 = on_Q_learn_v2.Q
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
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, max_N, expected_return)
	print("J_N of the online policy :")
	print(J_opt)


	## Fourth experimental protocol (bonus)
	print("\n--Fourth experimental protocol (bonus)--\n")

	# Initial epsilon_g
	epsilon_g = 1

	# Online Q-learning policy
	on_Q_learn_v4 = Online_Q_learning_v4(g, U, f_transition, n_traj, size_traj, start_state, learning_rate, gamma, 1)

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
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, max_N, expected_return)
	print("J_N of the online policy :")
	print(J_opt)