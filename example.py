# example usage of the algorithms presented in our paper
# executing this script on a Titan X takes about 6 minutes

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from givens import random_planted_matrix, elimination_factorization, greedy_baseline, right_givens
from util import symmetrized_norm
from givens_gpu import coordinate_descent_l1


d = 256
dlogd = int(d * np.log2(d))
n_planted = dlogd
n_givens  = d*(d-1)//2

U = random_planted_matrix(d, n_planted)

trajectories = []
trajectories.append(coordinate_descent_l1(U, n_givens))
trajectories.append(elimination_factorization(U, n_givens))
trajectories.append(greedy_baseline(U, n_givens))

methods = ['l1', 'structured_elimination', 'greedy_baseline']

data = []
interval = n_givens // 50
for method, trajectory in zip(methods, trajectories):
	V = np.eye(d)
	for k, t in enumerate(trajectory):
		c = math.cos(t[0])
		s = math.sin(t[0])
		V = right_givens(c, s, V, t[1], t[2])
		if k % interval == 0 or k == (len(trajectory)-1):
			approx_fro = symmetrized_norm(U, V)
			data.append([method, k/dlogd, approx_fro/np.sqrt(d)])

df = pd.DataFrame(data, columns=['method', 'iteration', 'approx_fro'])
sns.lineplot(x='iteration', y='approx_fro', hue='method', data=df)
plt.xlabel('$K / d\log_2(d)$')
plt.ylabel('$||U - \hat U||_{F,\mathrm{sym}} / \sqrt{d}$')
plt.savefig('givens_factorization.png')
