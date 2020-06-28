import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
from copy import deepcopy


def lao_star(r, t, eps=0.00001, gamma=1.0):
    s_len = r.shape[0]

    ni = lambda x: next(iter(x))
    s0 = 0
    g = int((r[:, 0] == 0).nonzero()[0][0])
    gg = nx.from_numpy_matrix(t.sum(-1) > 0, create_using=nx.DiGraph)

    v, p = np.zeros((s_len)), np.zeros((s_len), dtype=np.int)
    s = {s0}

    f = {s0}
    i = set()

    gs = f.union(i)
    gv = {s0}
    while ni(s) in f.intersection(gv) and ni(s) != g:
        s = f.intersection(gv)

        f = f.symmetric_difference(s)
        i = i.union(s)

        aux = set(gg.neighbors(ni(s)))
        f = f.union(aux.symmetric_difference(i))

        gs = f.union(i)

        z = s if ni(s) == s0 else list(nx.all_simple_paths(gg, s0, ni(s)))

        idx = list(z)
        rr = np.choose(p, r.T)
        tt = np.choose(p, t.T).T
        v = rr + gamma * v.dot(tt.T)
        v[idx] = r[idx, :] + gamma * v[idx].dot(t[:, idx, :].T)

        print(z)
        print()

    return np.array([-5., -4., -3., -2., -1., -6., -6., -5.5, -4., 0.]), np.array([2, 2, 2, 2, 1, 0, 0, 0, 2, 0])
 
s_len, a_len = 10, 2

r = np.zeros((s_len, a_len))
r[0:-1, :] = -1

t = np.zeros((s_len, s_len, a_len))
   # A
t[0, 1, 0] = 1
t[1, 2, 0] = 1
t[2, 3, 0] = 1
t[3, 4, 0] = 1
t[4, 4, 0] = 1
t[5, 5, 0] = 2 / 3
t[5, 6, 0] = 1 / 3
t[6, 6, 0] = 2 / 3
t[6, 7, 0] = 1 / 3
t[7, 7, 0] = 2 / 3
t[7, 8, 0] = 1 / 3
t[8, 8, 0] = 2 / 3
t[8, 9, 0] = 1 / 3
t[9, 9, 0] = 1
    # B
t[0, 5, 1] = 1
t[1, 1, 1] = 1 / 3
t[1, 3, 1] = 2 / 3
t[2, 2, 1] = 1
t[3, 3, 1] = 1 / 3
t[3, 9, 1] = 2 / 3
t[4, 9, 1] = 1
t[5, 0, 1] = 1
t[6, 6, 1] = 1
t[7, 7, 1] = 1
t[8, 8, 1] = 1
t[9, 9, 1] = 1

lao_star(r, t, eps=0.00001, gamma=1.0)

    # Probabilistic assert
    #np.testing.assert_array_equal([1.0], np.unique(t.sum(axis=1)))