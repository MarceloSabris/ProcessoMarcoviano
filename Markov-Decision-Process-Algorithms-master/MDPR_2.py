import math
import numpy as np
import GUI as gui
import math


def EvaluetaPolicy( V ,R,epsilon,S,A):
   res = epsilon +1
   while res>epsilon :
        #print(V)
        V_old = V.copy()
        for s in range(S): #para todos os estados 
            for a in range(A) : #para toda a ação 
                for sNext in range(S):  #para proxima ação 
                    Q[s,a] =  R(s) +  gamma*T[a,s,sNext]*V_old[sNext]
            V[s] = np.max(Q[s])
            P[s] = np.argmax(Q[s])
        res =0
        for s in range(S) :
            dif = abs(V_old[s] - V[s])
            if dif[0]>res : 
                res = dif[0]
   return  V,P

def ValueInterationWithRisk( V ,R,epsilon,S,A,Risk):
   res = epsilon +1
   while res>epsilon :
        #print(V)
        V_old = V.copy()
        Q = np.zeros((S,A))
        for s in range(S): #para todos os estados 
            for a in range(A) : #para toda a ação 
                for sNext in range(S):  #para proxima ação 
                    Q[s,a] = Q[s,a] +  ( (gamma*T[a,s,sNext]*V_old[sNext]))
                Q[s,a]=math.exp(-Risk*R(s))*Q[s,a]
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
P = np.ones((S,1))

epsilon = 0.00001
gamma = 1
#value interation 
Q = np.zeros((S,A))


#incia a polica com zeros 
#iniciar com -1 
#verificar lambda - sinal de lambda 
V = np.ones((S,1))   

conv = 0
P_oldpolicy = np.copy(P)
def res (aulas):
    if (aulas > 10):
       aulas=10
    return -2 -aulas
epsilon = 0.01
i = 1
while i<100 :  # a implementação de K> 0 será dentro do codigo 
    P_oldpolicy = np.copy(P)
    V,P = EvaluetaPolicyRisk( V ,res,epsilon,S,A,0.7)
    print(V)
    print(P)
    print(P_oldpolicy)
    if(np.array_equal(P_oldpolicy,P)):
       break
    i=i+1

print(V)
