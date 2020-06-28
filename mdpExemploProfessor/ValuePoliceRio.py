import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix

#como modelar meu mdp 
#% | 1 | 2 | 3 | 4 | 5|
#% | 6 | 7 | 8 | 9 |10|
#
S = 123  #quais são so estados 
A = 4  # cima 0, 1 baixo , 2 direita,3 esquerda 3  
T = np.zeros((A,S,S)) #matriz de transição é como a transição vai ocorrer AXSXS - p
#Acação , linha a posição que estou e coluna a probabilidade de ocorrer estado 
#cima
S1 = 3

def loadFile()
  
    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Action_Leste.txt', 'r') as f:
        data1 = f.readlines()

    r = np.genfromtxt(path.joinpath('Cost.txt'))
    r = -r[..., np.newaxis].repeat(len(act_name), axis=1)
    max_len = r.shape[0]

    arr = []
    for i in act_name:
        tmp = np.genfromtxt(path.joinpath('Action_{}.txt'.format(i)))
        row = tmp[:, 0].astype(int) - 1
        col = tmp[:, 1].astype(int) - 1
        val = tmp[:, 2]
        tmp = coo_matrix((val, (row, col)), shape=(max_len, max_len)).toarray()
        arr.append(tmp[..., np.newaxis])

    t = np.concatenate(arr, axis=-1)

    # Probabilistic assert
    np.testing.assert_array_equal([1.0], np.unique(t.sum(axis=1)))

    return r, t







def load_problem(num=1):
    act_name = ['Leste', 'Norte', 'Oeste', 'Sul']
    path = Path.home().joinpath('Ambientes', 'Ambiente{}'.format(num))

    r = np.genfromtxt(path.joinpath('Cost.txt'))
    r = -r[..., np.newaxis].repeat(len(act_name), axis=1)
    max_len = r.shape[0]

    arr = []
    for i in act_name:
        tmp = np.genfromtxt(path.joinpath('Action_{}.txt'.format(i)))
        row = tmp[:, 0].astype(int) - 1
        col = tmp[:, 1].astype(int) - 1
        val = tmp[:, 2]
        tmp = coo_matrix((val, (row, col)), shape=(max_len, max_len)).toarray()
        arr.append(tmp[..., np.newaxis])

    t = np.concatenate(arr, axis=-1)

    # Probabilistic assert
    np.testing.assert_array_equal([1.0], np.unique(t.sum(axis=1)))

    return r, t


def loadFiile ():
    S = 123 
    A = 4
    S1 = 3 
    T = np.full((A,S,S1),0,dtype=np.float) 
    C = np.full((S,A),-1,dtype=np.float) 

    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Action_Leste.txt', 'r') as f:
        data1 = f.readlines()
    result = []
    for line in data1:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    Shape = result.shape
    T[0][:S,:A]  = result[:S,:A]

    
    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Action_Norte.txt', 'r') as f:
        data2 = f.readlines()
    result = []
    for line in data1:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    Shape = result.shape
    #T[1][:Shape[0],:S]  = result[0][:Shape[0],:S] 
    T[1][:S,:A]  = result[:S,:A]
    
    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Action_Oeste.txt', 'r') as f:
        data3 = f.readlines()
    result = []
    for line in data3:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    Shape = result.shape
    T[2][:S,:A]  = result[:S,:A]
    
    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Action_Sul.txt', 'r') as f:
        data4 = f.readlines()

    result = []
    for line in data4:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    Shape = result.shape
    T[3][:S,:A]  = result[:S,:A]
    
    
    with open('C:\Marcovi\ProcessoMarcoviano\ProcessosInteracao\mdpExemploProfessor\Cost.txt', 'r') as f:
        custos = f.readlines()

    result = []
    for line in custos:
        splitted_data = line.split(' ')
        splitted_data = [item for item in splitted_data if item]
        splitted_data = [item.replace('E+', 'e') for item in splitted_data]

        result.append(splitted_data)
    result = np.array(result, dtype = 'float64')
    shape = result.shape
    #C[:shape[0],0] = result[:S,0]
    C[:S,0] = result[:S,0]
    C[:S,1] = result[:S,0]
    C[:S,2] = result[:S,0]
    C[:S,3] = result[:S,0]
    return T,C


T,R = loadFiile()

V = np.zeros((S,1))
P = np.zeros((S,1))
#print(V)
res = 1
dif =1
epsilon = 0.00001
gamma = 0.3
#value interation 
Q = np.zeros((S,A))
while res>epsilon :
   print(res)
   V_old = V.copy()

   for s in range(S): #para todos os estados 
      for a in range(A) : #para toda a ação 
          Q[s,a] = R[s,a]
          for sNext in range(S1):  #para proxima ação 
             Q[s,a] =  Q[s,a]  +  gamma*T[a,s,sNext]*V_old[sNext]
      V[s] = np.max(Q[s])
      P[s] = np.argmax(Q[s])
   res =0
   for s in range(S) :
       dif = abs(V_old[s] - V[s])
       if dif>res : 
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

