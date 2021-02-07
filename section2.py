import numpy as np
import math
from section1 import *

def expected_ret_det(g, U, x, my_policy, gamma, J_prev):

	u = my_policy.choose_action(x)

	return R(g, f_det(x, u, g.shape)) + gamma * J_prev[f_det(x, u, g.shape)]


def expected_ret_stoch(g, U, x, my_policy, gamma, J_prev):

	u = my_policy.choose_action(x)
	J_x_success = R(g, f_det(x, u, g.shape)) + gamma * J_prev[f_det(x, u, g.shape)]
	J_x_teleport =  R(g, (0,0)) + gamma * J_prev[0,0]

	return (J_x_success + J_x_teleport)/2

def compute_J_dyna(g, U, my_policy, gamma, N, expected_return):
	J = np.zeros(g.shape)

	for t in range(1, N+1):
		J_prev = J
		for k in range(J.shape[0]):
			for l in range(J.shape[1]):
				J[k, l] = expected_return(g, U, (k, l), my_policy, gamma, J_prev)

	return J

def compute_N(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma) / Br , gamma))

if __name__ == '__main__':

	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])
	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	gamma = 0.99

	x = (3,0)

	Br = g.max()
	thresh = 1
	N = compute_N(gamma, Br, thresh)

	print("Chosen N : " + str(N))

	my_policy = policy_cst(U, "up")
	expected_ret = expected_ret_det

	J = compute_J_dyna(g, U, my_policy, gamma, N, expected_ret)

	print(J)
