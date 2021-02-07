import numpy as np

from section1 import *
from section2 import *

# define pattern for MDP equivalent
class MDP_eq():
	# p(x_next | x, u)
	def p_transi(self, x_next, x, u):
		pass

	# r(x, u)
	def r_state_action(self, x, u):
		pass

class MDP_eq_det(MDP_eq):
		def __init__(self, g):
			self.g = g

		# p(x_next | x, u) of the equivalent MDP of the domain deterministic
		def p_transi(self, x_next, x, u):
			if f_det(x, u, self.g.shape) == x_next:
				return 1

			else:
				return 0

		# r(x, u) of the equivalent MDP of the domain deterministic
		def r_state_action(self, x, u):
			return self.g[f_det(x, u, self.g.shape)]

class MDP_eq_stoch(MDP_eq):
		def __init__(self, g):
			self.g = g

		# p(x_next | x, u) of the equivalent MDP of the domain stochastic
		def p_transi(self, x_next, x, u):
			if x_next == (0,0):
				return 0.5

			elif x_next == f_det(x, u, self.g.shape):
				return 0.5

			else:
				return 0

		# r(x, u) of the equivalent MDP of the domain stochastic
		def r_state_action(self, x, u):
			return 0.5 * self.g[f_det(x, u, self.g.shape)] + 0.5 * self.g[(0,0)]

# compute the value of Q for state x and action u,
# for t steps given all the value of Q for t-1 steps,
# for a given equivalent MDP with p(x_next | x, u) and r(x, u)
def Q_val(g, u, x, gamma, Q_prev, my_MDP_eq):

	# reward for going to the given state
	Q_val = my_MDP_eq.r_state_action(x, u)

	# consider all possible next states
	for k in range(Q_prev.shape[0]):
		for l in range(Q_prev.shape[1]):
			prob = my_MDP_eq.p_transi((k,l), x, u)
			# if action u can lead to that state (k, l),
			# consider the best possible reward afterward for t-1 steps
			if prob != 0:
				Q_val += gamma * prob * Q_prev[k, l, :].max()

	return Q_val

# compute the values of Q(x,u) for all states x and actions u with N steps,
# for a given equivalent MDP with p(x_next | x, u) and r(x, u)
def compute_Q_dyna(g, U, gamma, N, my_MDP_eq, get_min_N=False):
	Q = np.zeros(g.shape + (len(U),))

	# variables to look for smallest expected N
	min_N = N
	if get_min_N:
		curr_pol_mat = np.zeros(Q.shape[:2], dtype=np.int8)


	# for each step (dynamic ascending programming)
	for t in range(1, N):
		Q_prev = Q
		# compute for each state x and action u for t steps given the matrix for t-1 steps
		for k in range(Q.shape[0]):
			for l in range(Q.shape[1]):
				for u_idx in range(len(U)):
					Q[k, l, u_idx] = Q_val(g, U[u_idx], (k, l), gamma, Q_prev, my_MDP_eq)

		if get_min_N:
			# check if the policy has changed
			new_pol_mat = get_optimal_pol_mat(Q)
			if (new_pol_mat != curr_pol_mat).any():
				min_N = t
				curr_pol_mat = new_pol_mat

	return Q, min_N

def get_optimal_pol_mat(Q):

	policy_mat = np.zeros(Q.shape[:2], dtype=np.int8)

	# compute best action for each state (k,l)
	for k in range(Q.shape[0]):
		for l in range(Q.shape[1]):
			# best action is the one with the greater Q
			policy_mat[k,l] = np.argmax(Q[k,l])

	return policy_mat

# compute N such that the bound on the suboptimality
# for the approximation (over an horizon limited to N steps) of the optimal policy
# is smaller or equal to a given threshold
def compute_N_bis(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma)**2 / (2*Br) , gamma))

# policy class ruled by a matrix of state space size,
# U is the set of actions and pol_mat contains the index of the action to take for each state
class policy_set(cls_policy):
	def __init__(self, U, pol_mat):
		self.U = U
		self.pol_mat = pol_mat

	def choose_action(self, x):
		return U[self.pol_mat[x[0], x[1]]]

if __name__ == '__main__':

	# define problem's values
	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])
	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	gamma = 0.99
	x = (3,0)

	# compute N expected large enough
	Br = g.max()
	thresh = 0.1
	max_N = compute_N_bis(gamma, Br, thresh)
	print("Chosen N : " + str(max_N))
	print()

	# compute Q_N for all states x and actions u
	get_min_N = True
	my_MDP_eq = MDP_eq_det(g)

	Q, min_N = compute_Q_dyna(g, U, gamma, max_N, my_MDP_eq, get_min_N)
	print("Q_N function (u, x) :")
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

	if get_min_N:
		print("with N = " + str(min_N) + " as smallest N without changed afterward")
	print()

	# set the optimal policy and the kind of case considered (deterministic/stochastic)
	policy_Q = policy_set(U, policy_mat)
	expected_return = expected_ret_det

	# compute the expected returns (J)
	J_opt = compute_J_dyna(g, U, policy_Q, gamma, min_N, expected_return)
	print("J_N of the new policy :")
	if get_min_N:
		print("(with the smallest N = " + str(min_N) + ")")
	print(J_opt)
