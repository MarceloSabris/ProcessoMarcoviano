import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
import os
from scipy.sparse import coo_matrix
from pathlib import Path

def LoatTXT(Lote,Acoes):
    
    path = os.getcwd()
    path = path+"\Ambientes\Ambiente"+str(Lote) 
    pathCost = path + '\Cost.txt'

    with open(pathCost, 'r') as f:
        data1 = f.readlines()
    result = []
    for line in data1:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    shapeResul = result.shape
    S = shapeResul[0]
    C = np.full((shapeResul[0],len(Acoes)),1,dtype=np.float)   
    C[:S,0] = result[:S,0]
    C[:S,1] = result[:S,0]
    C[:S,2] = result[:S,0]
    C[:S,3] = result[:S,0]
  
    
   # retirado  do site https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html 
    for i in Acoes:
        Arquivo =  path + "\\" + i
        array =[]

 
    result = []
    for line in data1:
        with open(Arquivo, 'r') as f:
             data1 = f.readlines()
        result = []
        for line in data1:
            splitted_data = line.split(' ')
            splitted_data = [item for item in splitted_data if item]
            splitted_data = [item.replace('E+', 'e') for item in splitted_data]
            result.append(splitted_data)
        result = np.array(result, dtype = 'float64')
        linha = result[:, 0].astype(int) - 1
        coluna = result[:, 1].astype(int) - 1
        peso = result[:, 2]
        temporario = coo_matrix((peso, (linha, coluna)), shape=(S,S)).toarray()
        array.append(temporario[..., np.newaxis])

    t = np.concatenate(array, axis=-1)

    return C, t

Acoes = ['Action_Leste.txt', 'Action_Norte.txt', 'Action_Oeste.txt', 'Action_Sul.txt'] 
Lote = 1
R,T = LoatTXT(Lote,Acoes)



S = R.shape[0]  #quais são so estados 
A = R.shape[1]  # cima 0, 1 baixo , 2 direita,3 esquerda 3  
T = np.zeros((A,S,S)) #matriz de transição é como a transição vai ocorrer AXSXS - p
#Acação , linha a posição que estou e coluna a probabilidade de ocorrer estado 
#cima

V = np.zeros((S,1))
P = np.zeros((S,1))
#print(V)
res = 1
dif =1
epsilon = 0.00001
gamma = 1
#value interation 
Q = np.zeros((S,A))
while res>epsilon :
   print('oi',res)
   V_old = V.copy()

   for s in range(S): #para todos os estados 
      for a in range(A) : #para toda a ação 
          Q[s,a] = R[s,a]
          for sNext in range(S):  #para proxima ação 
             Q[s,a] =  Q[s,a]  +  gamma*T[a,s,sNext]*V_old[sNext]
      V[s] = np.max(Q[s])
      #P[s] = np.argmin(Q[s])
   res =0
   for s in range(S) :
       dif = abs(V_old[s] - V[s])
       if dif>res : 
           res = dif
              
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

