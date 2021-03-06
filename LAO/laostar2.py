import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.sparse import coo_matrix

costs = {0: 0 , 1:0 , 2:0, 3:4, 4:1,5:1, 6:2,7:0,8:0}
t = np.zeros((2,10,10 ), dtype=np.float)

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
    array =[]
    for i in Acoes:
        Arquivo =  path + "\\" + i
        
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


def GetLenA(t) :
    return t.shape[0]

def GetLenS(t):
    return t.shape[1]

def makeG(mg):
    for i in mg.edges:
        mg[i[0]][i[1]]['marked']=False
        mg[i[0]][i[1]]['andby']= None
    for i in mg.nodes:
        mg.nodes[i]['solved']=False
        mg.nodes[i]['cost']=0
        mg.nodes[i]['expanded']=True
     
def ResetMarks (mg) :
  for i in mg.edges:
        mg[i[0]][i[1]]['marked']=False
      

    

def Getcost(mg,node):
    return nx.get_node_attributes(mg,'cost')[node]

def solved(mg,node):
    return nx.get_node_attributes(mg,'solved')[node]


def andby(mg,edge1,edge2):
    return nx.get_edge_attributes(mg,'andby')[edge1,edge2]
    


def marked(mg,edge1,edge2):
    return nx.get_edge_attributes(mg,'marked')[edge1,edge2]

def findleaf(g):
    for i in g.nodes:
        g.remove_edges_from(nx.selfloop_edges(g))
        if (not g.__getitem__(i) )  :
            if i not in goalNodes:
                return i
    return None

def addToG(n,nodeList):
    for i in nodeList:
        # print(i)
        if not g.has_edge(n,i):
            g.add_edge(n,i,marked=False)
            g.nodes[i]['solved'] = False
            if i in goalNodes : 
                g.nodes[i]['solved'] = True



    

def ValueInteration(g,T,mainGraph): 
 ls = []
 res = 1
 lensG = len(g)
 passog = np.zeros(lensG,dtype=np.int)
 contador =0 
 for i in g:
    passog[contador] = i
    contador = contador +1   
 
 V = np.zeros(lensG,dtype=np.float)
 R = np.full(( lensG,GetLenA(T)),-1,dtype=np.float)
 P = np.full(( lensG),-1,dtype=np.int)
 epsilon = 0.00001
 gamma = 1
 #value interation 
 Q = np.zeros(( lensG,GetLenA(T)),dtype=np.float) 
 contador = 0
 while (res>epsilon and contador<100000 ) :
   contador = contador +1
   V_old = V.copy()
   index = 0 
   for s in passog: #para todos os estados 
      for a in range(GetLenA(T)) : #para toda a ação 
          Q[index,a] = R[index,a]
          contSnext = 0
          for sNext in passog:  #para proxima ação  
              Q[index,a] =  Q[index,a]  +  gamma*T[a,s,sNext]*V_old[contSnext]
              contSnext = contSnext + 1
      
      V[index] = np.min(Q[index])
      P[index] = np.argmin(Q[index])
      index = index +1
   res =0
   for s in range(lensG) :
       dif = abs(V_old[s] - V[s])
       if dif>res : 
           res = dif
 
 ResetMarks(g)
 g.remove_edges_from(nx.selfloop_edges(g))
 index = 0
 for i in passog:
       for n in list(g.neighbors(i)):
        
         if ((n != i and  mainGraph[i][n]['weight'] == P[int(np.where(passog == i)[0]) ]+1) or ( n != i and (sum(T[0,i]) == 0  or sum(T[1,i]) == 0  ) )) :
             if len(ls) > 0 : 
                 for edge in g.edges(ls[index-1]):
                     if edge[1] == n :
                        ls.append(n)
                        g[i][n]['marked']= True
                        index = index + 1 
             else : 
                 ls.append(n)
                 g[i][n]['marked']= True
                 index = index + 1 
             #g.nodes[i]['cost']=  T[a,int(i),n]
             #g.nodes[i]['expanded']= True
             #g[i][n]['andby'] = n
             #
               
                     
    #   if( V   ) 
 print(ls) 
 return [ls[index-1]]           


goalNodes = [9]
Acoes = ['Action_Leste.txt', 'Action_Norte.txt', 'Action_Oeste.txt', 'Action_Sul.txt'] 
Lote = 1
result =[]
R,T = LoatTXT(Lote,Acoes)



V = np.full(( GetLenS(t)),0,dtype=np.float)


mainGraph  = nx.from_numpy_matrix(np.array(T[0]), create_using=nx.DiGraph) #crio os nos fronteiras
labels = {i : i + 1 for i in mainGraph.nodes()}
pos = nx.circular_layout(mainGraph)
nx.draw_networkx_labels(mainGraph, pos, labels, font_size=15)
plt.show()


startNode = 5
makeG(mainGraph)
g = mainGraph.subgraph(startNode).copy() 





if startNode in goalNodes :
    g.node[startNode]['solved'] = True


while (solved(g,startNode) != True):
    #crio um grafo novo csó com n0
    gprime = g.subgraph(startNode).copy()
    #para cada vizinho do grafo
    for i in g.edges:
        #todos que ja foram marcados adiciono nos processados 
        if marked(g,i[0],i[1]):
            gprime.add_edge(i[0],i[1])

    g.remove_edges_from(nx.selfloop_edges(g))    
    n = findleaf(gprime)
    if (n == None):
       break
    #pego os vizinhos
    nodeList = list(mainGraph.neighbors(n))
    #add no g que o gravico que vai ficar com a fronteira   
    addToG(n,nodeList)
    print(g.nodes)
    s = [n]
    
    g.remove_edges_from(nx.selfloop_edges(g))
    while (len(s) != 0):
       
        #não é mais um no fronteira 
        for i in s:
            if i not in list(g.neighbors(i)):
                m = s.pop(s.index(i))
                break

        markList = ValueInteration(g,t,mainGraph)
        #for i in markList:
        #    g.edges[m,i]['marked'] = True
            #g.remove_edges_from(nx.selfloop_edges(g))
            #for j in list(g.neighbors(m)):
             #   if i != j:
              #      g.edges[m,j]['marked'] = False
               #     g.remove_edges_from(nx.selfloop_edges(g))
                #    for a in list(g.neighbors(j)):
                #        g.edges[j,a]['marked'] = False

        #checkNeighbors = True
        #for i in g.neighbors(m):
        #    if (i!=m and  marked(g,m,i)):
        #        if solved(g,i) == False:
        #            checkNeighbors = False  
        #if checkNeighbors:        
        #    g.nodes[m]['solved'] = True
        #atualizar custos 
        #if costs[m] != q[markList[0]]:
        #    costs[m] = q[markList[0]]
        #    for i in list(nx.predecessor(g,m)[m]):
        #        if (i!=m and  marked(g,m,i)):
        #            s.append(i)

        if solved(g,m):
            for i in list(nx.predecessor(g,m)[m]):
                if(i!=m and  marked(g,m,i)):
                    s.append(i)

markedge = nx.get_edge_attributes(g,'marked')


print("The answer graph contain below edges:")
for i in markedge:
    if markedge[i]:
        print(i)