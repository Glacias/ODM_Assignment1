import numpy as np

# apply an action to a state,
# keep current position if out of bound
def F(x, u, dim):
	return (min(max(x[0]+u[0], 0), dim[0]-1), min(max(x[1]+u[1], 0), dim[1]-1))

# give the reward by reaching a given state
def R(g, x_next):
	return g[x_next]

# deterministic transition from a state x applying an action u
def f_det(x, u, dim):
	return F(x, u, dim)

# stochastic transition (from the assignement) from a state x applying an action u,
# w computed directly inside
def f_stoch(x, u, dim):
	if np.random.rand() < 0.5:
		return F(x, u, dim)

	else:
		return (0,0)

# main class for creating a policy
class cls_policy():
	def choose_action(self, x):
		pass

# policy class for a constant direction
# give the U matrix and specify the direction desired
class policy_cst(cls_policy):
	def __init__(self, U, direction):
		self.U = U

		if direction == "down":
			self.action = U[0]

		elif direction == "up":
			self.action = U[1]

		elif direction == "right":
			self.action = U[2]

		else:
			self.action = U[3]

	def choose_action(self, x):
		return self.action

# policy class for a random action
class policy_rand(cls_policy):
	def __init__(self, U):
		self.U = U

	def choose_action(self, x):
		return self.U[np.random.randint(len(self.U))]

# apply a policy to find action and get the outgoing new state from f_transition
def get_next(g, U, x, my_policy, f_transition):
	u = my_policy.choose_action(x)
	x_next = f_transition(x, u, g.shape)

	return u, x_next

if __name__ == '__main__':
	# choose case : 0 for det and 1 for stoch
	case = 1

	# define problem's values
	g = np.array([[-3, 1, -5, 0, 19],
				[6, 3, 8, 9, 10],
				[5, -8, 4, 1, -8],
				[6, -9, 4, 19, -5],
				[-20, -17, -4, -3, 9]])

	U = [(1, 0), (-1, 0), (0, 1), (0, -1)]
	x = (3,0)

	# set the POLICY the the kind of TRANSITION expected (deterministic/stochastic)
	my_policy = policy_cst(U, "right")
	#my_policy = policy_rand(U)
	f_transition = [f_det, f_stoch][case]

	# Iterate for 10 actions
	print("Starting at " + str(x))
	for t in range(11):
		u, x_next = get_next(g, U, x, my_policy, f_transition)
		print("(x_" + str(t) + " = " + str(x) +
			", u_" + str(t) + " = " + str(u) +
			", r_" + str(t) + " = " + str(R(g, x_next)) +
			", x_" + str(t+1) + " = " + str(x_next) + ")")
		x = x_next