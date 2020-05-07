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
#valor interation 
Q = np.zeros((S,A))
#controle de interação 
iterations = 0 
is_value_changed=True
while is_value_changed: 
    is_value_changed = False
    iterations += 1 
    print("interação" , iterations)
    for s in range(S): 
        #rodando o value interation para caada linha 
        #utilizando o sum para somar todos os itens 
        # e for dentro do sum 
        V[s] = sum([T[int(P[s,0]),s,s1] * (R[s,int(P[s,0])] + gamma*V[s1]) for s1 in range(S)])
             #  sum([P[s,policy[s],s1] *   (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])
    for s in range(S): #para cada estado 
        q_best = V[s] #falo que o Q best é  valor do estado gerado pela interação de valor 
        # print "State", s, "q_best", q_best
        for a in range(A): #para cada ação 
            q_sa = sum([T[a,s, s1] * (R[s, a] + gamma * V[s1]) for s1 in range(S)]) #apliquei a melhora da politca para o estado e ação 
            if q_sa > q_best: # se o resultado for maior 
                print ("State", s, ": q_sa", q_sa, "q_best", q_best)
                P[s] = a
                q_best = q_sa
                is_value_changed=True
    

        
   
