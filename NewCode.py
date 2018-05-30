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
def usps_2classes(int1, int2, p) :
    index_train = np.concatenate((np.where(y_train==int1)[0], np.where(y_train==int2)[0]))
    index_test = np.concatenate((np.where(y_test==int1)[0], np.where(y_test==int2)[0]))   
    nb_tot = len(index_train) + len(index_test)
    nb_train = int(p*nb_tot)
    if(nb_train < len(index_train)):
        index_train = index_train[:nb_train].squeeze()
        index_test_add = index_train[nb_train:].squeeze()
        return x_train[index_train], y_train[index_train], np.concatenate((x_train[index_test_add], x_test)), np.concatenate((y_train[index_test_add], y_test))
    else:
        index_train_add = index_test[:nb_train-len(index_train)]
        index_test = index_test[nb_train-len(index_train):]
        return np.concatenate((x_train[index_train], x_test[index_train_add])), np.concatenate((y_train[index_train],y_test[index_train_add])), x_test[index_test], y_test[index_test]

x_train12, y_train12, x_test12, y_test12 = usps_2classes(1,2, 0.1)
state = np.concatenate((y_train12,np.array([0]*len(y_test12))))



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
    
def createAdj(x_train12, x_test12, sigma, yChapeau) :
    x = np.concatenate((x_train12,x_test12))
    n = x.shape[0]
    Adj = np.zeros((n,n))
    for i in range (n) :
        for j in range (n) :
            Adj[i][j] = sum((x[i] - x[j])**2)
    Adj = np.exp(-np.multiply(Adj,sigma))
    
    return Adj
 

def split(M,l,u):
    #Creation des sous-matrices
    M_ll=M[:l,:l]
    M_lu=M[:l,l:l+u]
    M_ul=M[l:l+u,:l]
    M_uu=M[l:l+u,l:l+u]
    return (M_ll, M_lu, M_ul, M_uu)


def labeliseG(Adj,states,l,u):
    d=np.sum(Adj,axis=1)
    D=np.diag(d)
    P=np.dot(np.linalg.inv(D),Adj)
    Adj_ll,Adj_lu,Adj_ul, Adj_uu=split(Adj,l,u)
    D_ll, D_lu, D_ul, D_uu = split(D,l,u)
    P_ll, P_lu, P_ul, P_uu = split(P,l,u)
    f_l=states[:l]
    #f_u=np.dot(np.linalg.inv(D_uu-Adj_uu),Adj_ul)
    I=np.eye(u)
    f_u=np.dot(np.linalg.inv(I-P_uu),P_ul)
    f_u=np.dot(f_u,f_l)
    return (np.concatenate((f_l,f_u)))

def plotErrorSigma(x_train12, x_test12, state) : 
    Adj = createAdj(x_train12,x_test12,0.1)
    l=len(x_train12)
    u=len(x_test12)
    score=[]
    test_sigma_log=np.arange(-2.2,0.1,0.1)
    test_sigma=10**(test_sigma_log)
    for sigma in test_sigma :
        Adj = createAdj(x_train12, x_test12, sigma)
        statesChapeau = labeliseG(Adj,states,l,u)
        statesChapeau = np.where(statesChapeau>1.5, 2, 1)
        print(sum(statesChapeau[len(y_train12):] == y_test12)/len(y_test12)*100, "%")    
        score.append(sum(statesChapeau[len(y_train12):] == y_test12)/len(y_test12)*100)
    plt.figure()
    plt.plot(test_sigma_log,score)
    plt.xlabel("log sigma")
    plt.ylabel("Score")
    plt.title("Score en fonction de sigma")
    
def plotErrorP(x_train12, x_test12, state) : 
    Adj = createAdj(x_train12,x_test12,0.1)
    l=len(x_train12)
    u=len(x_test12)
    score=[]
    test_sigma_log=np.arange(-2.2,0.1,0.1)
    test_sigma=10**(test_sigma_log)
    for sigma in test_sigma :
        Adj = createAdj(x_train12, x_test12, sigma)
        statesChapeau = labeliseG(Adj,states,l,u)
        statesChapeau = np.where(statesChapeau>1.5, 2, 1)
        print(sum(statesChapeau[len(y_train12):] == y_test12)/len(y_test12)*100, "%")    
        score.append(sum(statesChapeau[len(y_train12):] == y_test12)/len(y_test12)*100)
    plt.figure()
    plt.plot(test_sigma_log,score)
    plt.xlabel("log sigma")
    plt.ylabel("Score")
    plt.title("Score en fonction de sigma")

    
#if False : # Test 2
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
    

    
   


