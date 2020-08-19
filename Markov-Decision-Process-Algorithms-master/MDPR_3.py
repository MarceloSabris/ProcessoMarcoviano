import math
import numpy as np
import GUI as gui


def policy_iteration(T,S,A,R, gamma = 0.9, epsilon = 0.01, optimal_value = None):
   # Initialization
    V = [0]*S
    pol = [0]*S
    old_V = V
    vs = []
    n_iter = 0
    v_iter = 0
    while True:
        # Policy improvement
        policy_stable = True
        for s in range(A):
            old_action = pol[A]
            possibilities = [sum(T[s,a,k] *(R(k) + gamma * V[k]) for k in range(S)) for a in range(A)]
            pol[s] = max(enumerate(possibilities)))
            if(old_action != pol[s]):
                policy_stable = False
        if policy_stable:
            print ("Converged in %d iterations, %d evaluations" % (n_iter, v_iter))
            return V, pol, n_iter
        n_iter += 1


def EvaluetaPolicy( V ,R,epsilon,S,A):
   res = epsilon +1
   while res>epsilon :
        #print(V)
        V_old = V.copy()
        for s in range(S): #para todos os estados 
            for a in range(A) : #para toda a ação 
                for sNext in range(S):  #para proxima ação 
                    Q[s,a] =  R(a+s) +  gamma*T[a,s,sNext]*V_old[sNext]
                    V[s] = np.max(Q[s])
                    P[s] = np.argmax(Q[s])
        res =0
        for s in range(S) :
            dif = abs(V_old[s] - V[s])
            if dif[0]>res : 
                res = dif[0]
   return  V,P

def Problem(nx,ny,na):
    '''
    :param nx: number of states in the x axis
    :param ny: number of states in the y axis
    :param na: number of actions
    '''
    'define a value'
    T = np.zeros((na,nx,ny))

    x=0
    for a in range(na):  # each state
        for y in range(ny -1) :
            x = a+y
            if (x>10): 
                  x=10
            T[a,y,nx-1] = 0.08*x + 0.02*a
            T[a,y,x] = 1- T[a,y,nx-1]
    return T
S = 12
A = 5

# Test Script
T = Problem(S, S,A)
P = np.zeros((S,1))

epsilon = 0.00001
gamma = 1
#value interation 
Q = np.zeros((S,A))


#incia a polica com zeros 
#V = np.zeros((S,1))   
#conv = 0
#P_oldpolicy = np.copy(P)
def res (aulas):
    if (aulas > 10):
       aulas=10
    return -2 -aulas
#epsilon = 0.00001
#while True:  # a implementação de K> 0 será dentro do codigo 
#    P_oldpolicy = np.copy(P)
#    V,P = EvaluetaPolicy( V ,res,epsilon,S,A)
#    print(V)
#    print(P)
#    print(P_oldpolicy)
#    if(np.array_equal(P_oldpolicy,P)):
#       break

policy_iteration(T,S,A,res)