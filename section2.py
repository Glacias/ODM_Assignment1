import numpy as np
import math
from section1 import *

def expected_ret_det(g, U, x, policy, gamma, N):
	if N <= 0:
		return 0

	else:
		u = policy(U)
		return R(g, f_det(x, u, g.shape)) + gamma * expected_ret_det(g, U, f_det(x, u, g.shape), policy, gamma, N-1)


def expected_ret_stoch(g, U, x, policy, gamma, N):
	if N <= 0:
		return 0

	else:
		u = policy(U)
		J_x_success = R(g, f_det(x, u, g.shape)) + gamma * expected_ret_det(g, U, f_det(x, u, g.shape), policy, gamma, N-1)
		J_x_teleport =  R(g, (0,0)) + gamma * expected_ret_det(g, U, (0,0), policy, gamma, N-1)
		return (J_x_success + J_x_teleport)/2

def compute_N(gamma, Br, thresh):
	return math.ceil(math.log(thresh * (1-gamma) / Br , gamma))

def compute_J(g, U, policy, gamma, N, expected_return):
	J = np.zeros(g.shape)

	for k in range(J.shape[0]):
		for l in range(J.shape[1]):
			J[k, l] = expected_return(g, U, (k, l), policy, gamma, N)

	return J

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

	"""
	J_mu_x_det = expected_ret_det(g, U, x, policy_right, gamma, N)
	J_mu_x_stoch = expected_ret_stoch(g, U, x, policy_right, gamma, N)

	print(J_mu_x_det)
	print(J_mu_x_stoch)
	"""

	J = compute_J(g, U, policy_right, gamma, N, expected_ret_det)

	print(J)
