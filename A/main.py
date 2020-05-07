class No():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0 #distanci do no atual para o inicio
        self.h = 0 #heristica para estimar a distancia 
        self.f = 0 #total do custo f= g+h

    def __eq__(self, other):
        return self.position == other.position 
def AAlgoritmo (quenaointendi, inicio, fim):
    
    #inicia o inicio e o fim do node 
    noInicio = No(None, start)
    noInicio.g = noInicio.h = noInicio.f = 0
    noFim = No(None, end)
    noFim.g = nofim.h = noFim.f = 0
    
    #inicia os nos abaerto e finais como null 
    nos_abertos = []
    nos_fechados = []

    # Inicia os nós abertos com o inicial 
    nos_abertos.append(noInicio)

    while len(open_list) > 0: 
        #pega o no inicial 
        No_atual = open_list[0]
        Indice_atual = 0
        
        #pego cara indice e verifico 
        #se o custo do indice atual for maior que o proximo 
        #fico com o atual 
        for index, item in enumerate(open_list):
            if item.f < nota_atual.f:
                current_node = item
                current_index = index
