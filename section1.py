import numpy as np

def F(x, u, dim):
	return (min(max(x[0]+u[0], 0), dim[0]-1), min(max(x[1]+u[1], 0), dim[1]-1))

def R(g, x_next):
	return g[x_next]

def f_det(x, u, dim):
	return F(x, u, dim)

def f_stoch(x, u, dim):
	if np.random.rand() < 0.5:
		return F(x, u, dim)

	else:
		return (0,0)

def policy_rand(U):
	return U[np.random.randint(len(U))]

def policy_right(U):
	return (0, 1)

def get_next(g, U, x, policy, f_transition):
	u = policy(U)
	x_next = f_transition(x, u, g.shape)

	return u, x_next

if __name__ == '__main__':

	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])

	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	x = (3,0)

	print("Starting at " + str(x))
	for t in range(10):
		u, x_next = get_next(g, U, x, policy_right, f_stoch)
		print("(x_" + str(t) + " = " + str(x) +
			", u_" + str(t) + " = " + str(u) +
			", r_" + str(t) + " = " + str(R(g, x_next)) +
			", x_" + str(t+1) + " = " + str(x_next) + ")")
		x = x_next