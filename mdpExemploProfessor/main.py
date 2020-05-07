import numpy as np
import matplotlib.pyplot as plt 
#como modelar meu mdp 
#% | 1 | 2 | 3 | 4 | 5|
#% | 6 | 7 | 8 | 9 |10|
#
S = 10  #quais são so estados 
A = 4  # cima 0, 1 baixo , 2 direita,3 esquerda 3  
T = np.zeros((A,S,S)) #matriz de transição é como a transição vai ocorrer AXSXS - p
#Acação , linha a posição que estou e coluna a probabilidade de ocorrer estado 
#cima



T[0] =    [[  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
           [  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
           [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
           [  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
           [  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [  0,0.5,  0,  0,  0,0.5,  0,  0,  0,  0],#rio ver o problema inicial , por isso é 50%
           [  0,  0,0.5,  0,  0,0.5,  0,  0,  0,  0],#rio ver o problema inicial , por isso é 50%
           [  0,  0,  0,0.5,  0,0.5,  0,  0,  0,  0],#rio Ver o problema inicial,por isso é 50%
           [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]]

#baixo -> ver sempre o rio 

T[1] =      [[  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
            [  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
            [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
            [  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0,0.5,0.5,  0,  0,  0],
            [  0,  0,  0,  0,  0,0.5,  0,0.5,  0,  0],
            [  0,  0,  0,  0,  0,0.5,  0,  0,0.5,  0],
            [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]]

#direita
T[2] =      [[  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
            [  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
            [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [  0,  0,  0,  0,  0,0.5,  0,0.5,  0,  0],
            [  0,  0,  0,  0,  0,0.5,  0,  0,0.5,  0],
            [  0,  0,  0,  0,  0,0.5,  0,  0,  0,0.5],
            [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]] 

#esquerda 
T[3] =      [[1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
              [0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
              [0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
              [0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
              [0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
              [0,  0,  0,  0,  0,0.5,0.5,  0,  0,  0],
              [0,  0,  0,  0,  0,0.5,  0,0.5,  0,  0],
              [0,  0,  0,  0,  0,  0,  0,  0,  0,  1]]


R = np.full((S,A),-1)    #recompensa  

R[9] = [0,0,0,0] #estado meta 


V = np.zeros((S,1))
P = np.zeros((S,1))
#print(V)
res = 1
epsilon = 0.00001
gamma = 1
#value interation 
Q = np.zeros((S,A))
while res>epsilon :
   print(V)
   V_old = V.copy()

   for s in range(S): #para todos os estados 
      for a in range(A) : #para toda a ação 
          Q[s,a] = R[s,a]
          for sNext in range(S):  #para proxima ação 
             Q[s,a] =  Q[s,a] +  gamma*T[a,s,sNext]*V_old[sNext]
      V[s] = np.max(Q[s])
      P[s] = np.argmax(Q[s])
   res =0
   for s in range(S) :
       dif = abs(V_old[s] - V[s])
       if dif[0]>res : 
           res = dif[0]
              
print(P)

#Police interarion

""" def EvaluetaPolicy( V ,res,epsilon,S,A):
   while res>epsilon :
        #print(V)
        V_old = V.copy()
        for s in range(S): #para todos os estados 
            for a in range(A) : #para toda a ação 
                Q[s,a] = R[s,a]
                for sNext in range(S):  #para proxima ação 
                    Q[s,a] =  Q[s,a] +  gamma*T[a,s,sNext]*V_old[sNext]
                    V[s] = np.max(Q[s])
                    P[s] = np.argmax(Q[s])
        res =0
        for s in range(S) :
            dif = abs(V_old[s] - V[s])
            if dif[0]>res : 
                res = dif[0]
   return  V,P
#incia a polica com zeros 
V = np.zeros((S,1))   
conv = 0
P_oldpolicy = np.copy(P)
while True:  # a implementação de K> 0 será dentro do codigo 
    P_oldpolicy = np.copy(P)
    V,P = EvaluetaPolicy( V ,res,epsilon,S,A)
    print(V)
    print(P)
    print(P_oldpolicy)
    if(np.array_equal(P_oldpolicy,P)):
       break
 """

