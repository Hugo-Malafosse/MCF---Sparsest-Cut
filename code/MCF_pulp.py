import csv
import itertools
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value

plt.rcParams['figure.figsize'] = [20, 14]


def setup(path_demande, path_noeud, path_edge):

    file = open(path_edge)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()

    EDGES = []
    for row in rows:
        row1 = row[0].split(';')
        row2 = []
        for s in row1:
            row2.append(int(s))

        EDGES.append(row2)

    file2 = open(path_demande)
    csvreader2 = csv.reader(file2)
    header2 = next(csvreader2)
    rows2 = []
    for row in csvreader2:
        rows2.append(row)

    file2.close()

    DEMANDE = []
    for row in rows2:
        rowA = row[0].split(';')
        rowB = []
        for s in rowA:
            rowB.append(float(s))

        DEMANDE.append(rowB)

    file1 = open(path_noeud)
    csvreader1 = csv.reader(file1)
    header1 = next(csvreader1)
    rows1 = []
    for row in csvreader1:
        rows1.append(row)

    file.close()

    NODES = []
    for row in rows1:

        row3 = row[0].split(';')
        row4 = []
        for s in row3:
            row4.append(s)

        NODES.append(row4)

    return NODES, EDGES, DEMANDE, len(EDGES), len(NODES), len(DEMANDE)


NODES, EDGES, DEMANDE, nombre_edge, nombre_noeuds, nombre_demande = setup("../graphe normal/demande.csv", "noeux.csv", "edge.csv")

#print(EDGES)

#print('nombre edges : ', nombre_edge)
#print('nombre noeuds : ', nombre_noeuds)
#print('nombre demandes : ', nombre_demande)


def create_nx(weight):
    g = nx.Graph()
    for i in range(nombre_noeuds):
        g.add_node(i, label=NODES[i][2], col='blue')

    for i in range(nombre_edge):
        g.add_edge(EDGES[i][1], EDGES[i][2], weight=weight[EDGES[i][0] - 1], styl='solid')
    return g


def afficher_graphe(weight, highlights):

    G1 = create_nx(weight)
    for i in range(nombre_noeuds):
        G1.add_node(i, label=NODES[i][2], col='blue')

    for i in range(nombre_edge):
        G1.add_edge(EDGES[i][1], EDGES[i][2], styl='solid', col='black')

    pos = {i: (float(NODES[i][5]), float(NODES[i][4]))
           for i in range(nombre_noeuds)}

    liste = list(G1.nodes(data='col'))

    colorNodes = {}
    for noeud in liste:
        if noeud[0] in highlights:
            colorNodes[noeud[0]] = 'red'
        else:
            colorNodes[noeud[0]] = noeud[1]

    colorList = [colorNodes[node] for node in colorNodes]

    labels_edges = {edge: G1.edges[edge]['weight'] for edge in G1.edges}

    liste = list(G1.nodes(data='label'))
    labels_nodes = {}
    for noeud in liste:
        labels_nodes[noeud[0]] = noeud[1]

    # nodes
    #print(pos)
    nx.draw_networkx_nodes(G1, pos, node_size=100, node_color=colorList, alpha=0.9)

    colorEdge = []
    for i in range(nombre_edge):
        colorEdge.append('black')

    nx.draw_networkx_labels(G1, pos, labels=labels_nodes,
                            font_size=15,
                            font_color='black',
                            font_family='sans-serif')
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels_edges, font_color='red')
    # edges
    nx.draw_networkx_edges(G1, pos, width=3, edge_color=colorEdge)
    plt.axis('off')
    plt.savefig('fig1.png')

    plt.show()
    return G1


#weight1 = [1 for i in range(nombre_edge)]
#afficher_graphe(weight1, [])


def create_path(demande):
    weight = [1 for _ in range(nombre_edge)]
    G1 = create_nx(weight)
    return nx.all_simple_paths(G1, DEMANDE[demande][0], DEMANDE[demande][1])


def find_edge(node1, node2):
    for i in range(nombre_edge):
        if (EDGES[i][1] == node1 and EDGES[i][2] == node2) or (EDGES[i][2] == node1 and EDGES[i][1] == node2):
            return i


def find_ze():
    # Parametres

    D = [DEMANDE[i][2] for i in range(nombre_demande)]
    Capacite = [EDGES[i][4] for i in range(nombre_edge)]

    # Définition des variables

    Z = []
    for i in range(nombre_edge):
        Z.append(LpVariable('z'+str(i), lowBound=0))

    Y = []
    for i in range(nombre_demande):
        Y.append(LpVariable('y'+str(i), lowBound=0))

    Lp_prob = LpProblem('Problem', LpMinimize)
    Lp_prob += sum([Z[i]*Capacite[i] for i in range(nombre_edge)])

    for demand in range(nombre_demande):
        print(demand/nombre_demande)
        paths = create_path(demand)
        for path in paths:

            expr = 0
            for i in range(len(path)-1):
                node1 = path[i]
                node2 = path[i+1]
                edge = find_edge(node1, node2)
                expr += Z[edge]

            expr = expr >= Y[demand]

            Lp_prob += expr

    Lp_prob += sum([D[i]*Y[i] for i in range(nombre_demande)]) >= 1

    status = Lp_prob.solve()  # Solver
    #print(LpStatus[status])

    VALUEZ = [value(Z[i]) for i in range(nombre_edge)]
    return VALUEZ, value(Lp_prob.objective)


def create_matrix():
    VALUEZ, valueopt = find_ze()
    weight = VALUEZ
    G = create_nx(weight)
    matr = list(nx.shortest_path_length(G, weight='weight'))
    matrbis = []
    for elt in matr:
        eltbis = [0 for i in range(nombre_noeuds)]
        for subelt in elt[1]:
            eltbis[subelt] = elt[1][subelt]
        matrbis.append(eltbis)

    return matrbis


def partition(random1):
    logn = int(np.log(nombre_noeuds))
    ENS = []
    s = [i for i in range(nombre_noeuds)]
    for i in range(1, int(np.log(nombre_noeuds)) + 1):
        if random1 and logn < math.comb(nombre_noeuds, 2**i):
            ensemble = []
            while len(ensemble) < logn:
                petitens = []
                while len(petitens) < 2**i:
                    ajout = np.random.choice(s)
                    if ajout not in petitens:
                        petitens.append(ajout)
                ensemble.append(petitens)

        else:
            ensemble = list(itertools.combinations(s, 2 ** i))

        ENS.append(ensemble)

    return ENS


def compute_distorsion(vects):
    M = create_matrix()
    R = []
    for i in range(nombre_noeuds):
        for j in range(nombre_noeuds):
            if i != j:
                dij = M[i][j]
                nij = np.linalg.norm(vects[i]-vects[j], ord=1)
                if nij != 0:
                    R.append(dij/nij)
                else:
                    R.append(np.infty)
    return max(R)


def compute_plongement(random1):
    ENS = partition(random1)
    vects = []
    M = create_matrix()
    s = [i for i in range(nombre_noeuds)]
    for j in range(nombre_noeuds):
        for i in range(len(ENS)):
            ensemble = ENS[i]
            vect = []
            card = 2**i
            for ens in ensemble:
                dist = min([M[j][p] for p in ens])
                vect.append(dist/(int(np.log(nombre_noeuds))*math.comb(nombre_noeuds, card)))

        vects.append(vect)
    print('VECTS ::::: ', vects)
    num = 0
    denum = 0

    sums = []

    for ind in range(len(vects[0])):

        for ind1 in range(nombre_edge):
            num += EDGES[ind1][4]*np.abs(vects[EDGES[ind1][1]][ind] - vects[EDGES[ind1][2]][ind])

        for h in range(nombre_demande):

            denum += DEMANDE[h][2]*np.abs(vects[int(DEMANDE[h][0])][ind] - vects[int(DEMANDE[h][1])][ind])
        if denum>0:
            sums.append(num/denum)
        else:
            sums.append(np.infty)
    min_dim = np.argmin(sums)
    print('min dim :::', min_dim)
    vectsbis = [[vects[i][min_dim],i] for i in range(len(vects))]

    dtypes = [('vecteur', float), ('node', int)]
    vectsort = sorted(vectsbis, key=lambda x:  x[0])
    vectsortbis = [vectsort[i][1] for i in range(len(vectsort))]

    return vectsortbis


def find_sp(random1):
    nodes = compute_plongement(random1)

    sc = []
    for i in range(len(nodes)-1):
        set = nodes[:i+1]

        demandsum = 0
        capsum = 0
        for dem in DEMANDE:
            if (dem[0] not in set and dem[1] in set) or (dem[0] in set and dem[1] not in set):

                demandsum += dem[2]

        for edge in EDGES:
            if (edge[1] not in set and edge[2] in set) or (edge[1] in set and edge[2] not in set):
                capsum += edge[4]
        if demandsum >0:
            sc.append(capsum/demandsum)
        else:
            sc.append(np.infty)
    print("CCCUUUTS :::: ", sc)
    return nodes[:sc.index(min(sc))+1], min(sc)


weight1 = [1 for i in range(nombre_edge)]
sc_random, sc = find_sp(True)
#sc_normal = find_sp(False)
#print('SPARSEST CUT NORMAL::::: ', sc_normal)
print('SPARSEST CUT RANDOM ::::: ', sc_random, sc)

#afficher_graphe(weight1, sc_normal)
afficher_graphe(weight1, sc_random)




'''vects = []
M = create_matrix()
s = [i for i in range(nombre_noeuds)]
for j in range(nombre_noeuds):
    for i in range(1, int(np.log(nombre_noeuds))+1):
        ensemble = list(itertools.combinations(s, 2**i))
        vect = []
        card = 2**i
        for ens in ensemble:
            dist = min([M[j][p] for p in ens])
            vect.append(dist/(int(np.log(nombre_noeuds))*math.comb(nombre_noeuds, card)))

    vects.append(vect)
for h in range(nombre_demande):
    print('DEMANDE' + str(h))
    print(vects[int(DEMANDE[h][0])])
    print(vects[int(DEMANDE[h][1])])'''


