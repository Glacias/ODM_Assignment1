import numpy as np
import math
from section1 import *

# compute the expected return for x for t steps given all the expected returns for t-1 steps,
# for a deterministic transition on a given policy
def expected_ret_det(g, U, x, my_policy, gamma, J_prev):

	u = my_policy.choose_action(x)

	return R(g, f_det(x, u, g.shape)) + gamma * J_prev[f_det(x, u, g.shape)]

# compute the expected return for x for t steps given all the expected returns for t-1 steps,
# for the stochastic transition (from the assignement) on a given policy
def expected_ret_stoch(g, U, x, my_policy, gamma, J_prev):

	u = my_policy.choose_action(x)

	# return for w < 0.5 (action succeed as in deterministic case)
	J_x_success = R(g, f_det(x, u, g.shape)) + gamma * J_prev[f_det(x, u, g.shape)]
	# return for w >= 0.5 (action failed, teleported to (0, 0) state)
	J_x_teleport =  R(g, (0,0)) + gamma * J_prev[0,0]

	# expectation over both case (equal probabilities)
	return (J_x_success + J_x_teleport)/2

# compute the expected return J(x) for all states x with N steps,
# for a given expactation case (deterministic/stochastic) and a given policy
def compute_J_dyna(g, U, my_policy, gamma, N, expected_return):
	J = np.zeros(g.shape)

	# for each step (dynamic ascending programming)
	for t in range(1, N+1):
		J_prev = J
		# compute for each state for t steps given the matrix for t-1 steps
		for k in range(J.shape[0]):
			for l in range(J.shape[1]):
				J[k, l] = expected_return(g, U, (k, l), my_policy, gamma, J_prev)

	return J

# compute N such that the infinite norm of the error on J is smaller or equal to a given threshold
def compute_N(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma) / Br , gamma))

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

	# compute appropriate N
	Br = g.max()
	thresh = 1
	N = compute_N(gamma, Br, thresh)

	print("Chosen N : " + str(N))

	# set the policy and the kind of case considered (deterministic/stochastic)
	my_policy = policy_cst(U, "right")
	expected_ret = expected_ret_det

	# compute the expected returns (J)
	J = compute_J_dyna(g, U, my_policy, gamma, N, expected_ret)

	print(J)
