import numpy as np
import section1 as s1

def expected_ret_det(g, U, x, gamma, N):
	if N == 0:
		return 0

	else:
		u = s1.policy_rand(U)
		return s1.R(g, x, u) + gamma * expected_ret_det(g, U, s1.get_next_rand(g, U, x)[1], gamma, N-1)



if __name__ == '__main__':

	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])
	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	gamma = 0.99

	x = (3,0)

	N = 0
	tresh = 0.0001
	gamma_N = 1
	while gamma_N > tresh:
		gamma_N *= gamma
		N += 1

	print("Chosen N : " + str(N))

	J_mu_x = expected_ret_det(g, U, x, gamma, N)

	print(J_mu_x)