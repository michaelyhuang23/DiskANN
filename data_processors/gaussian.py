import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

assert(len(sys.argv) >= 3)

cluster_n = int(sys.argv[1])
approx_n = int(sys.argv[2])

cluster_sizes = []

with open('../data/gaussian.data', 'w') as f:
	for i in range(cluster_n):
		mx = np.random.uniform(0, 10)
		my = np.random.uniform(0, 10)
		n = np.random.randint(20, approx_n // cluster_n * 2)
		print(n)
		cluster_sizes.append(n)
		xs = np.random.normal(mx, 1, n)
		ys = np.random.normal(my, 1, n)
		lines = [f'{str(x)} {str(y)}\n' for x, y in zip(xs, ys)]
		f.writelines(lines)

with open('../data/gaussian.gt', 'w') as f:
	for i, cluster_size in enumerate(cluster_sizes):
		f.writelines([f'{str(i+1)}\n' for j in range(cluster_size)])

