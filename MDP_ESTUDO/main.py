import numpy as np
import matplotlib.pyplot as plt 

probs = np.array([
        [0, 0.3, 0.4, 0.3, 0],   # s_0 -> s' 
        [0, 0.0, 0.4, 0.3, 0.3], # s_1 -> s'
        [0, 0.2, 0.0, 0.7, 0.1], # s_2 -> s'
        [0, 0.1, 0.1, 0.0, 0.8], # s_3 -> s'
        [0, 0, 0, 0, 0]     # s_4 -> s'
    ])

R = np.array([
        [0, -1, -1, -1, 0], # R(s_0) -> s'
        [0, 0, -1, -1, 1],  # R(s_1) -> s'
        [0, -1, 0, -1, 1],  # R(s_2) -> s'
        [0, -1, -1, 0, 1],  # R(s_3) -> s'
        [0, 0, 0, 0, 0]
    ])

v = np.zeros(probs.shape[0])
v_old = v.copy()
gamma = 0.9
delta = 1e-5
delta_t = 1
dif = 1
#print(v)
#print(v_old)
while delta_t > delta:
    for s in range(len(probs)):
        v[s] = np.sum([
                probs[s][sp1] * (R[s][sp1] + gamma * v_old[sp1])
                for sp1 in range(len(probs[s]))
            ])
    ##print(v)
    delta_t = np.sum(np.abs(v - v_old))
    v_old = v.copy()
#print(v)
#consequentimente s3 é amel