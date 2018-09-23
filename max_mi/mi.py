import numpy as np
from scipy.special import gamma as g


def h(x):
	shape = x.shape
	d = shape[1]*1.
	N = shape[0]
	e = 0.00001
	
	rho = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
			if i == j:
				continue
			else:
				rho[i, j] = np.linalg.norm(x[i,:] - x[j,:])

	rho = np.sort(rho, axis=0)
	rho_min = rho[1, :]
	
	h = d * np.average(rho)
	h = h + np.log((N-1)*np.pi**(d/2) / g(1 + d/2))
	h = h + 0.577
	return h

if __name__ == '__main__':
	x = np.random.random((5, 20))
	print h(x)