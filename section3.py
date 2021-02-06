import numpy as np

from section1 import *
from section2 import *

def p_det(x_next, x, u, shape):
	if f_det(x, u, shape) == x_next:
		return 1

	else:
		return 0

def p_stoch(x_next, x, u, shape):
	if x_next == (0,0):
		return 0.5

	elif x_next == f_det(x, u, shape):
		return 0.5

	else:
		return 0

def r_det(x, u, g):
	return g[f_det(x, u, g.shape)]

def r_stoch(x, u, g):
	return 0.5 * g[f_det(x, u, g.shape)] + 0.5 * g[(0,0)]


def Q_val(g, u, x, gamma, Q_prev, p_func, r_func):

	Q_val = r_func(x, u, g)

	# see slide 24
	for k in range(Q_prev.shape[0]):
		for l in range(Q_prev.shape[1]):
			prob = p_func((k,l), x, u, Q_prev.shape[:2])
			if prob != 0:
				Q_val += gamma * prob * Q_prev[k, l, :].max()

	return Q_val


def compute_Q_dyna(g, U, policy, gamma, N, p_func, r_func):
	Q = np.zeros(g.shape + (len(U),))
	print(Q.shape)

	for t in range(1, N):
		Q_prev = Q
		for k in range(Q.shape[0]):
			for l in range(Q.shape[1]):
				for u_idx in range(len(U)):
					Q[k, l, u_idx] = Q_val(g, U[u_idx], (k, l), gamma, Q_prev, p_func, r_func)

	return Q

def policy_Q(U, Q, x):
	#print(Q[(1,1)])
	return U[np.argmax(Q[x])]

if __name__ == '__main__':

	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])
	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	gamma = 0.99

	x = (3,0)

	# N computation from section2, should not be used here
	Br = g.max()
	thresh = 1
	N = compute_N(gamma, Br, thresh)

	print("Chosen N : " + str(N))

	Q = compute_Q_dyna(g, U, policy_up, gamma, N, p_det, r_det)

	print(np.moveaxis(Q, 2, 0))

	#print(policy_Q(U, Q, (1,1)))

	print("policy :")
	policy_mat = np.zeros(g.shape, dtype=np.int8)
	#instruction = ["down  ", "up    ", "right ", "left  "]
	instruction_arrow = ['\u2193', '\u2191', '\u2192', '\u2190']
	for k in range(g.shape[0]):
		for l in range(g.shape[1]):
			policy_mat[k,l] = np.argmax(Q[k,l])
			print(instruction_arrow[policy_mat[k,l]], end="")
		print()


	#print(policy_mat)