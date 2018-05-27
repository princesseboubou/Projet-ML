## Import des bibliothèques nécessaires
import networkx as nx # Bibliothèques pour les graphes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from random import *
from math import *
import matplotlib.image as mpimg
from operator import add
import pickle



## Données USPS
def load_usps(filename) :
    with open(filename ,"r") as f:
        f.readline()
        data = [ [float(x) for x in l.split()] for l in f if len(l.split())>2] 
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data) : 
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    
# load data
fileroot = "/Users/marieheurtevent/Desktop/Ponts/MachineLearning/Projet/USPS/USPS"
x_test, y_test = load_usps(fileroot + '_test.txt')
x_train, y_train = load_usps(fileroot + '_train.txt')
    
def visual_ex() :
    rand_tab1 = np.array([randint(0,1000)%x_train.shape[0] for i in range (2)])
    rand_tab2 = np.array([randint(0,1000)%x_test.shape[0] for i in range (2)])
    
    plt.figure()
    plt.subplot(2,2,1)
    show_usps(x_train[rand_tab1[0]])
    plt.subplot(2,2,2)
    show_usps(x_train[rand_tab1[1]])
    plt.subplot(2,2,3)
    show_usps(x_test[rand_tab2[0]])
    plt.subplot(2,2,4)
    show_usps(x_test[rand_tab2[1]])
    plt.show()
    
    print(y_train[rand_tab1[0]],y_train[rand_tab1[1]],y_test[rand_tab2[0]],y_test[rand_tab2[1]])
    
visual_ex()


# on ne prend que les images de 2 classes différentes
def usps_2classes(int1, int2) :
    index_train = np.sort(np.concatenate(( np.where(y_train==int1)[0], np.where(y_train==int2)[0] )))
    index_test = np.sort(np.concatenate(( np.where(y_test==int1)[0], np.where(y_test==int2)[0] )))
    return x_train[index_train], y_train[index_train], x_test[index_test], y_test[index_test]
    
x_train12, y_train12, x_test12, y_test12 = usps_2classes(1,2)



################# Début de notre travail expérimental #########################


## Premières fonctions
# Affichage selon l'état des noeuds
def affichage(G, int1, int2):
    """ Affiche le graphe G en utilisant différentes couleurs selon l'état des noeuds
        Vert ("1"), bleu (sans label), rouge ("2")"""
        
    # Tableaux des noeuds dans chaque état
    state1 = [d[0] for d in G.nodes(data=True) if d[1]['state'] == int1]
    state2 = [d[0] for d in G.nodes(data=True) if d[1]['state'] == int2]
    stateWO = [d[0] for d in G.nodes(data=True) if d[1]['state'] == 0]

    # On s'assure que rien n'est affiché
    plt.clf()
    # Postionnement des noeuds
    pos = nx.spring_layout(G)
    # Affichage des noeuds
    nx.draw_networkx_nodes(G, pos, nodelist=state1, node_color='g', label=int1)
    nx.draw_networkx_nodes(G, pos, nodelist=state2, node_color='r', label=int2)
    nx.draw_networkx_nodes(G, pos, nodelist=stateWO, node_color='b', label="sans label")
    # Affichage des liens
    nx.draw_networkx_edges(G, pos)
    # Affichage des labels
    nx.draw_networkx_labels(G, pos, labels=dict(zip(list(G.nodes()),list(G.nodes()))), fontsize = 8)
    # Légende
    plt.legend()
    # Affichage du nombre de noeuds dans chaque état
    print("Nombre de ", int1, " = ", len(state1))
    print("Nombre de ", int2, " = ", len(state2))
    print("Nombre sans label = ", len(stateWO))
    
    plt.show()
  

# Create graph    
def createG(Adj,states) : 
    G = nx.from_numpy_matrix(Adj)
    nx.set_node_attributes(G, dict(zip(G.node(),list(states))),'state')
    return G
    
def createAdj(x_train12, x_test12, sigma) :
    x = np.concatenate((x_train12,x_test12))
    n = x.shape[0]
    Adj = np.zeros((n,n))
    for i in range (n) :
        for j in range (n) :
            Adj[i][j] = np.dot(x[i],x[j]) # somme du produit des intensites des pixels correspondants
    Adj = np.exp(-np.multiply(Adj,sigma))
    return Adj
    

# Fonctions qui changent l'état des noeuds d'un graphe
def labeliseG(Adj, states):
    """Labelise les noeuds non labelisés"""
    new_states = states.copy() # Dictionnaire des nouveaux états
        
    new_states[states==0] = np.argmin([np.sum((Adj[np.where(states==0)]).T[np.where(states==i)].T,axis=1) for i in range (1,3)],axis=0) +1 # sera à changer quand int1, int2 != 1, 2
    # G = createG(Adj, new_states) # On impose à G les nouveaux états
    return new_states

    
if True : # Test 2
    Adj = createAdj(x_train12,x_test12,0.1)
    #states = np.array([1,1,1,1,1,1,1,2,1,2,0,0,0,0,0,0,0,0,0,0])
    states = np.concatenate((y_train12,np.array([0]*len(y_test12))))
    #G = createG(Adj,states)
    #plt.figure()
    #affichage(G,1,2)
    statesChapeau = labeliseG(Adj,states)
    print(sum(statesChapeau[len(y_train12):] == y_test12)/len(y_test12)*100, "%")
    #plt.figure()
    #affichage(G,1,2)
    


if False: # Test 1
    Gtest = nx.Graph()
    Gtest.add_node(0, state=1)
    Gtest.add_node(1, state=2)
    Gtest.add_node(2, state=0)
    Gtest.add_node(3, state=0)
    Gtest.add_node(4, state=1)
    Gtest.add_node(5, state=0)
    Gtest.add_node(6, state=1)
    Gtest.add_node(7, state=2)
    Gtest.add_weighted_edges_from([(0,1,0.5), (0,2,0.5), (1,3,0.5), (2,7,0.5), (5,4,0.5), (1,6,0.5), (3,7,0.5), (2,5,0.5), (3,4,0.5)])
    print(Gtest.nodes(data=True))
    state = nx.get_node_attributes(Gtest, 'state')
    affichage(Gtest,1,2)
    
    





##












