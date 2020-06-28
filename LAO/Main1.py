import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

import networkx as nx
import pydot as pgv # need pygraphviz or pydot for nx.to_agraph()

G = nx.DiGraph()
G.add_edge(1,2,weight=7)
G.add_edge(2,3,weight=8)
G.add_edge(3,4,weight=1)
G.add_edge(4,1,weight=11)
G.add_edge(1,3)
G.add_edge(2,4)



def lao_star(r, t, eps=0.00001, gamma=1.0):

    s_len = r.shape[0]

    
    ni = lambda x: next(iter(x))
    
    s0 = 0 # iniciando os conjuntos 
    g = int((r[:, 0] == 0).nonzero()[0][0]) # pego  a solução  
    gg = nx.from_numpy_matrix(t.sum(-1) > 0, create_using=nx.DiGraph) #crio os nos fronteiras 

    v, p = np.zeros((s_len)), np.zeros((s_len), dtype=np.int) #seto com 0 V e p 
    s = {s0}

    f = {s0}#zero os valores 
    i = set() # seto igual a null os nos espandidos 

    gs = f.union(i) #faço a união dos conjunto , pq ? 
    gv = {s0}
    while ni(s) in f.intersection(gv) and ni(s) != g:
        s = f.intersection(gv) #pego so estados que não sao metas 

        f = f.symmetric_difference(s) #tiro ele das solução 
      
        i = i.union(s) # faço uma uniao com S para incluir ele no interiror 

          #F ← F ∪ {x /∈ I : ∃a ∈ A T(s, a, x) > 0} ???? Tirei os nos, ele não esta na fronteira
        aux = set(gg.neighbors(ni(s)))
        f = f.union(aux.symmetric_difference(i))

        gs = f.union(i) # atualizo meu G 

        z = s if ni(s) == s0 else list(nx.all_simple_paths(gg, s0, ni(s))) #eu tiro os nos que não dependen de S
        
        #listo os nos adiasentes 
        idx = list(z)
        rr = np.choose(p, r.T)
        tt = np.choose(p, t.T).T
        v = rr + gamma * v.dot(tt.T)
        v[idx] = r[idx, :] + gamma * v[idx].dot(t[:, idx, :].T)
        #Reconstrua GˆV ?

#s0
#sobre estados de Gˆs0
        print(z)
        print()

    return np.array([-5., -4., -3., -2., -1., -6., -6., -5.5, -4., 0.]), np.array([2, 2, 2, 2, 1, 0, 0, 0, 2, 0])





s_len, a_len = 10, 10

r = np.zeros((s_len, a_len))
r[0:-1, :] = -1

t = np.zeros((s_len, a_len))

# A matriz (l,C)
t[0, 1] = 1
t[1, 2] = 1
t[2, 3] = 1
t[3, 4] = 1
t[4, 4] = 1
t[5, 5] = 2 / 3
t[5, 6] = 1 / 3
t[6, 6] = 2 / 3
t[6, 7] = 1 / 3
t[7, 7] = 2 / 3
t[7, 8] = 1 / 3
t[8, 8] = 2 / 3
t[8, 9] = 1 / 3
t[9, 9] = 1


s=     [[1,  1 ,0  ,0   ,0   ,1,0   ,0   ,0   ,0],
        [1, 1/3,1  ,2/3 ,0   ,0,0   ,0   ,0   ,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]]


G = nx.from_numpy_matrix(np.array(s), create_using=nx.MultiDiGraph())



nx.draw(G, with_labels=True, arrows = True, connectionstyle='arc3, rad = 0.1')
nx.draw_circular(G)
# labels = {i : i + 1 for i in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels, font_size=15)
plt.show()

# B
#t[0, 5, 1] = 1
#t[1, 1, 1] = 1 / 3
#t[1, 3, 1] = 2 / 3
#t[2, 2, 1] = 1
#t[3, 3, 1] = 1 / 3
#t[3, 9, 1] = 2 / 3
#t[4, 9, 1] = 1
#t[5, 0, 1] = 1
#t[6, 6, 1] = 1
#t[7, 7, 1] = 1
#t[8, 8, 1] = 1
#t[9, 9, 1] = 1

#testa 
# Probabilistic assert
#np.testing.assert_array_equal([1.0], np.unique(t.sum(axis=1)))
# iniciando os conjuntos 


#lao_star(r,t)



