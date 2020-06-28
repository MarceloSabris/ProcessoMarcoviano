import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
from copy import deepcopy
a_len = 2
R = np.full((10,a_len),-1,dtype=np.float)    #recompensa  
interacao = lambda x: next(iter(x))
R[9] = [0,0] #estado meta 


def ValueInteration(Z,T,V,R,A,S): 
 res = 1
 #V = np.zeros((S,1))
 P = np.zeros((S,1))
 epsilon = 0.0001
 gamma = 1
 doing = set()
 #value interation 
 Q = np.zeros((S,A),dtype=np.float) 
 contador = 0
 while (res>epsilon and contador<10000 ) :
   contador = contador +1
   V_old = V.copy()

   for s in Z: #para todos os estados 
      for a in range(A) : #para toda a ação 
          Q[s,a] = R[s,a]
          for sNext in range(S):  #para proxima ação  
              Q[s,a] =  Q[s,a]  +  gamma*T[a,s,sNext]*V_old[sNext]
      
      V[s] = np.max(Q[s])
      P[s] = np.argmax(Q[s])
   res =0
   for s in range(S) :
       dif = abs(V_old[s] - V[s])
       if dif[0]>res : 
           res = dif[0]
   
   
 return V,P              
   





    
#0->R -> 2
#1->S -> 3
#2-> V -> 4
MGrafo1=  [ [0  ,1    ,0  ,0   ,0   ,2  ,0   ,0   ,0   ,0], #1
            [0  , 2   ,1  ,2   ,0   ,0  ,0   ,0   ,0   ,0], #2
            [0  ,0    ,0  ,1   ,0   ,0  ,0   ,0   ,0   ,0], #3 
            [0  ,0    ,0  ,2   ,1   ,0  ,0   ,0   ,0   ,2], #4
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,2], #5
            [2  ,0    ,0  ,0   ,0   ,1  ,1   ,0   ,0   ,0], #6 
            [0  ,0    ,0  ,0   ,0   ,0  ,1   ,1   ,0   ,0], #7 
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,1   ,1   ,0], #8
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,1   ,1],  #9
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0]] 
#Grafo = nx.from_numpy_matrix(np.array(MGrafo1), create_using=nx.DiGraph) #crio os nos fronteiras

MGrafo=  [[0  ,  1  ,0  ,0   ,0   ,1  ,0   ,0   ,0   ,0], #1
        [0 , 1/3 ,1  ,2/3  ,0   ,0  ,0   ,0   ,0   ,0], #2
        [0  , 0   ,0 , 1   ,0   ,0  ,0   ,0   ,0   ,0], #3 
        [0  ,0    ,0  ,1/3  ,1   ,0  ,0   ,0   ,0   ,2/3], #4
        [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,1], #5
        [1  ,0    ,0  ,0   ,0   ,2/3 ,1/3   ,0   ,0   ,0], #6 
        [0  ,0    ,0  ,0   ,0   ,0  ,2/3   ,1/3   ,0   ,0], #7 
        [0  ,0    ,0  ,0   ,0   ,0  ,0   ,2/3  ,1/3   ,0], #8
        [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,2/3   ,1/3],  #9
        [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0     ,0 ]] 
s_len, a_len = 10, 2
t = np.zeros((a_len,s_len,s_len ), dtype=np.float)


t[0] =    [ [0  ,1    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #1
            [0  ,0    ,1  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #2
            [0  ,0    ,0  ,1   ,0   ,0  ,0   ,0   ,0   ,0], #3 
            [0  ,0    ,0  ,0   ,1   ,0  ,0   ,0   ,0   ,0], #4
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #5
            [0  ,0    ,0  ,0   ,0   ,2/3,1/3 ,0   ,0   ,0], #6 
            [0  ,0    ,0  ,0   ,0   , 0 ,2/3 ,1/3 ,0   ,0], #7 
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,2/3 ,1/3   ,0], #8
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,2/3   ,1/3], #9
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0]] 

t[1] =    [ [0  ,0    ,0  ,0   ,0   ,1  ,0   ,0   ,0   ,0], #1
            [0  ,1/3  ,0  ,2/3 ,0  ,0  ,0   ,0   ,0   ,0], #2
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #3 
            [0  ,0    ,0  ,1/3 ,0   ,0  ,0   ,0   ,0   ,2/3], #4
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #5
            [1  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #6 
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #7 
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0], #8
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0],  #9
            [0  ,0    ,0  ,0   ,0   ,0  ,0   ,0   ,0   ,0]] 


# t[0, 0, 1] = 1
# t[1, 0, 5 ] = 1



# t[0, 1, 2] = 1
# t[1, 1, 3] = 2/3
# t[1, 1, 1] = 1/3

# t[0, 2, 3] = 1



# t[0, 3, 4] = 1
# t[1, 3, 9] = 1



# t[0, 4, 9] = 1

# t[0, 5, 6] = 1/3
# t[2, 5 , 5] = 2/3
# t[1, 5, 0 ] = 1

# t[0, 6, 7] = 1/3
# t[2, 6 , 6] = 2/3

# t[0, 7, 8] = 1/3
# t[2, 7 , 7] = 2/3



# t[0, 8, 9] = 1/3
# t[2, 8 ,8 ] = 2/3



#t[4, 4, 0] = 1
#t[5, 5, 0] = 2 / 3
#t[5, 6, 0] = 1 / 3
#t[6, 6, 0] = 2 / 3
#t[6, 7, 0] = 1 / 3
#t[9, 9, 0] = 1 / 3
#t[7, 8, 0] = 1 / 3
#t[8, 8, 0] = 2 / 3
#t[8, 9, 0] = 1 / 3
#t[9, 9, 0] = 1
    # B

Grafo = nx.from_numpy_matrix(np.array(MGrafo1), create_using=nx.DiGraph) #crio os nos fronteiras
pos = nx.circular_layout(Grafo)
nx.draw_circular(Grafo)
labels = {i : i + 1 for i in Grafo.nodes()}
nx.draw_networkx_labels(Grafo, pos, labels, font_size=15)
plt.show()
#epsilon = 0.001
#gamma = 1


v, p = np.zeros((s_len)), np.zeros((s_len), dtype=np.int) #seto com 0 V e p
v = np.full(s_len,1) 
s0 = 5 # inicio o no que vai começar 

#r = np.zeros((s_len,a_len ), dtype=np.float)
#r[0:-1, :] =-1 ## o 10 é minha solução 

g = int((R[:, 0] == 0).nonzero()[0][0]) # pego que é a solução no meu prolema
sAtual=5
s={sAtual} 
f= {sAtual} #fronteira
i = set() #no fronteira igual a nada  #nos que ja foram espandidos i inteiror
gs = f.union(i) #g do passo  Todos os nos que foram expandidos mais a fronteira 
gv = {sAtual} #gv igual a so 
z = []
Q = np.zeros((10,a_len))
#VisaoGeral = np.array[10,2]
Sneighbors = [10]

nivel = 1
z.append([sAtual])
V = np.full((s_len,1),0, dtype=np.float)
Custo = np.full((3,10),None)
Custo[1,0] = 0 
Custo[0,0] = sAtual
Custo[2,0] = sAtual
contadorVizinho =nivel
#v[5] = 
while interacao(s) in f.intersection(gv) and interacao(s) != g: 
   Solucao = np.full((10),None) 
   Solucao[0] = 5
  
   s = f.intersection(gv).intersection([sAtual]) # nó que estou espandindo 
   f = f.symmetric_difference(s) #nao é mais um no fronteira 
   i = i.union(s) #incluo ele no interior 
   #pegos todos os vizinhos  
   neighbors = set(Grafo.neighbors(interacao(s))) #pego todos os vizinhos pra execução em lote
   f = f.union(neighbors.symmetric_difference(i)) #tiro os processados  
   gs = i.union(f) #atualizo o gs
   #z=np.zeros((a_len,s_len,s_len),  dtype=np.float)
   # if sAtual in neighbors:
   # neighbors.remove(sAtual)
  
   for neighbor in neighbors:
      if neighbor != sAtual :
         Custo[0,contadorVizinho]  = int(neighbor)
         contadorVizinho= contadorVizinho+1
      if [neighbor] not in z:
           z.append([neighbor])
   #z.sort()
   # z=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
   #nivel = 9
   V,P =ValueInteration(z,t,V,R,a_len,s_len)
   
   count =0
   #for Node in z:
       # for neighbor in Grafo.neighbors(Node[0]) :
        #   if Grafo.edges[(Node[0],neighbor)]['weight']==P[Node[0]]+2 :
        #      Solucao[count] = int(neighbor)
        #      count = count+1
   continua = True
   while ( continua ) : 
       escolhido =  np.argmax(t[int(P[sAtual][0]),int(Solucao[count])] )
       for valor in range(s_len):
           if (Custo[1,valor] ==  escolhido):
               Custo[1,valor] = np.max(t[int(P[sAtual][0]),int(Solucao[count])] )
       for cont in range(contadorVizinho): 

       Solucao[count+1] = np.argmax(t[int(P[Solucao[count]][0]),int(Solucao[count])] )
       
       #for neighbor in Grafo.neighbors(Solucao[count]):
        #   T[P[Solucao[count]],Solucao[count],:]
         # if Grafo.edges[(Solucao[count],neighbor)]['weight']==P[Solucao[count]]+1 :
          #    if Solucao[count+1]  == None :
           #      Solucao[count+1] = int(neighbor)
            #  elif t[(Solucao[count],neighbor)]['weight'] >  Grafo.edges[(Solucao[count+1],neighbor)]['weight'] :
             #    Solucao[count+1] = int(neighbor)
                     
       if(nivel == count+1):
            continua=False 
       count = count +1

   proximo = Solucao[nivel] 
   print(Solucao) 
   nivel = nivel +1

   #a = set
   #print(len(neighbors))
   #pego os vizinhos 
   # vou na matriz orinal 
   # e pego só os que estão dentro do caminho , pois devem ser alterados 
   # depois eu atualizo a original com os caminhos alterados 
   # no for ja trato seelf loops 
   
   
   
   gama = 1
   #como eu vou pegar o v os outros passos já estão la 
   #while i< 2 :
   #for neighbor in neighbors:
    #        print(neighbor)
     #       for interior in i:
     #           for paths in list(nx.all_shortest_paths(Grafo,interior, neighbor)) :
                       
     #               print(paths)
     #               Node =  len(paths)-1
                    #pOrigemAnt = paths[Node -1]
                    #pDestinoAnt = paths[Node]
                    #passoAnterior =  Grafo[pOrigemAnt][pDestinoAnt]["weight"] -2
      #              while( Node>0):
       #              for CountCam in range( len(paths)-1):
        #                pOrigem  = paths[Node -1]
         #               pDestino = paths[Node]
                        #passo =  Grafo[pOrigem][pDestino]["weight"] -2
                        #Grafo[pOrigem][pDestino]["weight"] -2
                        #print('Porigem',pOrigem,'pDestino',pDestino, 'Passo',  passo)
          #              dest.append([pDestino])
                        #pOrigemAnt = pOrigem
                        #passoAnterior =  passo
                        
           #          Node = Node-1
           # Sneighbors[interior].append(dest)       


   #for cont in range(10):
    #    v[cont] = np.max(Q[cont])
    
   #passo = np.argmax(Q[sAtual])+2
   
   #print('pass0',passo)
   #for neighbor in neighbors:
    #    if Grafo.edges[(sAtual,neighbor)]['weight']==passo and i != sAtual:
    #       proximo = neighbor       
       # V[sAtual] =  min(Q[SAtual])     
                        #Grafo[sAtual][node]["weight"] -2   
                        #t[Grafo[sAtual][node]["weight"] -2,sAtual,node]
                #for a in range(a_len) :
                  # z.add([sAtual,c])
                   #z[a,sAtual,c] = t[a,sAtual,c]
                 #  listupdate.append([ path[c],path[c+1] ])
   #pAtual = 0 
   #vValue = ValueInteration(s_len,a_len,z,T) 
   #for neighbor in neighbors:
    #   if (Grafo[sAtual][neighbor]['weight'] == vValue[sAtual])+2:
     #      pAtual =  vValue[sAtual])
      #     proximo = neighbor
      #     break;   
    #v[stual] = t[pAtual,sAtual,proximo]
   #i = s_len -2 
   #while i<0 : 
    #  v[i] = V[i] + v[i+1] 
    #  i=1-1 

   #  if 

   #Grafo[5][6]['weight']


  
   #v[sAtual] = 1
   


   #Q1 =  np.zeros((s_len,s_len), dtype=np.float)
   #dif=True
   
  # while dif :
   #print(v)
   #V_old = deepcopy(v)  
   
 #  for s1 in range(s_len): #para todos os estados 
  #      for a in range(s_len):  
  #         # Q1[s,a] = r[s,a]   
  #          for sNext in range(s_len):  #para proxima ação 
  #              Q1[s1,a] =   v[s1] +   gamma*z[s1,sNext]*V_old[sNext]
  #      v[s1] = min(Q1[s1])
  
  
   s.remove(sAtual)     
   sAtual = proximo 
   
   s.add(sAtual)  
   gv.add(interacao(neighbors)) 
   if sAtual not in neighbors:
     gv.add(sAtual)
     f.add(sAtual)
   if sAtual not in gv:
     gv.add(sAtual)
   if sAtual not in f:
     f.add(sAtual)

   print(s)
   

        

   
   



