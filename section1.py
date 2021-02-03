import numpy as np

def F(x, u, dim):
	return (min(max(x[0]+u[0], 0), dim[0]-1), min(max(x[1]+u[1], 0), dim[1]-1))

def R(g, x, u):
	return g[F(x, u, g.shape)]

def f_det(x, u, dim):
	return F(x, u, dim)

def f_stoch(x, u, dim):
	if np.random.rand() < 0.5:
		return f_det(x, u, dim)

	else:
		return (0,0)

def policy_rand(U):
	return U[np.random.randint(len(U))]

def get_next_rand(g, U, x, t):
	u = policy_rand(U)
	x_next = f_det(x, u, g.shape)
	print("(x_" + str(t) + " = " + str(x) +
		", u_" + str(t) + " = " + str(u) +
		", r_" + str(t) + " = " + str(R(g, x, u)) +
		", x_" + str(t+1) + " = " + str(x_next) + ")")

	return x_next

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
		x = get_next_rand(g, U, x, t)