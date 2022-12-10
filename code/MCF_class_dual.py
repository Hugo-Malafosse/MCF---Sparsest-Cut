import csv
import itertools
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pulp import *
plt.rcParams['figure.figsize'] = [20, 14]


class MCF_graph_bis:

    def setup(self, path_demande, path_noeud, path_edge):
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

    def __init__(self, path_demande, path_noeud, path_edge):

        NODES, EDGES, DEMANDE, nombre_edge, nombre_noeuds, nombre_demande = self.setup(path_demande, path_noeud, path_edge)

        self.DEMANDE = DEMANDE
        self.NODES = NODES
        self.EDGES = EDGES
        self.nombre_demande = nombre_demande
        self.nombre_noeuds = nombre_noeuds
        self.nombre_edge = nombre_edge

    def create_nx(self, weight):
        g = nx.Graph()
        for i in range(self.nombre_noeuds):
            g.add_node(i, label=self.NODES[i][2], col='blue')

        for i in range(self.nombre_edge):
            g.add_edge(self.EDGES[i][1], self.EDGES[i][2], weight=weight[self.EDGES[i][0] - 1], styl='solid')
        return g

    def afficher_graphe(self, weight, highlights):

        G1 = self.create_nx(weight)
        for i in range(self.nombre_noeuds):
            G1.add_node(i, label=self.NODES[i][2], col='blue')

        for i in range(self.nombre_edge):
            G1.add_edge(self.EDGES[i][1], self.EDGES[i][2], styl='solid', col='black')

        pos = {i: (float(self.NODES[i][5]), float(self.NODES[i][4]))
               for i in range(self.nombre_noeuds)}

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
        # print(pos)
        nx.draw_networkx_nodes(G1, pos, node_size=100, node_color=colorList, alpha=0.9)

        colorEdge = []
        for i in range(self.nombre_edge):
            colorEdge.append('black')

        nx.draw_networkx_labels(G1, pos, labels=labels_nodes,
                                font_size=15,
                                font_color='black',
                                font_family='sans-serif')
        nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels_edges, font_color='red')
        # edges
        nx.draw_networkx_edges(G1, pos, width=3, edge_color=colorEdge)
        plt.axis('off')


        plt.show()
        return G1

    def export_graph(self, weight, highlights, left, right):
        G1 = self.create_nx(weight)
        for i in range(self.nombre_noeuds):
            G1.add_node(i, label=self.NODES[i][2], col='blue')

        for i in range(self.nombre_edge):
            G1.add_edge(self.EDGES[i][1], self.EDGES[i][2], styl='solid', col='black')

        pos = {i: (float(self.NODES[i][5]), float(self.NODES[i][4]))
               for i in range(self.nombre_noeuds)}

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
        # print(pos)
        nx.draw_networkx_nodes(G1, pos, node_size=100, node_color=colorList, alpha=0.9)

        colorEdge = []
        for i in range(self.nombre_edge):
            colorEdge.append('black')

        nx.draw_networkx_labels(G1, pos, labels=labels_nodes,
                                font_size=15,
                                font_color='black',
                                font_family='sans-serif')
        nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels_edges, font_color='red')
        # edges
        nx.draw_networkx_edges(G1, pos, width=3, edge_color=colorEdge)
        plt.axis('off')
        plt.savefig(f"./files_test/graphs_drawn/bipartite graph with {left} left nodes and {right} right nodes.png")
        plt.clf()
    # weight1 = [1 for i in range(nombre_edge)]
    # afficher_graphe(weight1, [])

    def create_path(self, demande):
        weight = [1 for _ in range(self.nombre_edge)]
        G1 = self.create_nx(weight)
        return nx.all_simple_paths(G1, self.DEMANDE[demande][0], self.DEMANDE[demande][1])

    def find_edge(self,  node1, node2):
        for i in range(self.nombre_edge):
            if (self.EDGES[i][1] == node1 and self.EDGES[i][2] == node2) or (self.EDGES[i][2] == node1 and self.EDGES[i][1] == node2):
                return i

    def find_ye(self):
        # Parametres

        D = self.DEMANDE.copy()
        Capacite = [self.EDGES[i][4] for i in range(self.nombre_edge)]

        # Définition des variables

        Z = []
        Y = []
        for i in range(self.nombre_edge):
            Y.append(LpVariable('y' + str(i), lowBound=0))

        for h in range(self.nombre_demande):
            zh = []
            for j in range(self.nombre_noeuds):
                zh.append(LpVariable('z' + str(h)+str(j)))
            Z.append(zh)
        Lp_prob = LpProblem('Problem', LpMinimize)
        Lp_prob += lpSum([Y[i] * Capacite[i] for i in range(self.nombre_edge)])

        Lp_prob += lpSum([D[h][2]*(Z[h][int(D[h][0])] - Z[h][int(D[h][1])]) for h in range(self.nombre_demande)]) >= 1
        grande_expr = []
        for i in range(self.nombre_edge):
            y = Y[i]
            print(y)
            for h in range(self.nombre_demande):

                expr0 = Z[h][int(self.EDGES[i][1])] - Z[h][int(self.EDGES[i][2])]

                expr1 = expr0 <= y
                expr2 = -expr0 <= y

                print(expr1)
                print(expr2)

                Lp_prob += expr1
                Lp_prob += expr2

        status = Lp_prob.solve()  # Solver
        print('status ::::: ', LpStatus[status])

        VALUEZ = [value(Y[i]) for i in range(self.nombre_edge)]
        print('Values :::: ', VALUEZ)
        return VALUEZ, value(Lp_prob.objective)

    def create_matrix(self):
        VALUEZ, valueopt = self.find_ye()
        weight = VALUEZ
        G = self.create_nx(weight)
        matr = list(nx.shortest_path_length(G, weight='weight'))
        matrbis = []
        for elt in matr:
            eltbis = [0 for i in range(self.nombre_noeuds)]
            for subelt in elt[1]:
                eltbis[subelt] = elt[1][subelt]
            matrbis.append(eltbis)

        return matrbis

    def partition(self, random1):
        logn = int(np.log(self.nombre_noeuds))
        ENS = []
        s = [i for i in range(self.nombre_noeuds)]
        i = 1
        while 2**i <= self.nombre_noeuds:
            if random1 and logn < math.comb(self.nombre_noeuds, 2 ** i):
                ensemble = []
                while len(ensemble) < logn:
                    petitens = []
                    while len(petitens) < 2 ** i:
                        ajout = np.random.choice(s)
                        if ajout not in petitens:
                            petitens.append(ajout)
                    ensemble.append(petitens)

            else:
                ensemble = list(itertools.combinations(s, 2 ** i))

            ENS.append(ensemble)
            i += 1
        return ENS

    def compute_distorsion(self, vects):
        M = self.create_matrix()
        R = []
        for i in range(self.nombre_noeuds):
            for j in range(self.nombre_noeuds):
                if i != j:
                    dij = M[i][j]
                    nij = np.linalg.norm(np.array(vects[i]) - np.array(vects[j]), ord=1)
                    if nij != 0:
                        R.append(dij / nij)
                    else:
                        R.append(0)
        return max(R)

    def compute_plongement(self, random1):
        ENS = self.partition(random1)
        #print('ENSSSS ///// ', ENS)
        vects = []
        M = self.create_matrix()
        for j in range(self.nombre_noeuds):
            vect = []
            for i in range(len(ENS)):

                ensemble = ENS[i]

                #print(ensemble)

                card = 2 ** i
                for ens in ensemble:
                    #print(ens)
                    dist = min([M[j][p] for p in ens])
                    vect.append(dist / (int(np.log(self.nombre_noeuds)) * math.comb(self.nombre_noeuds, card)))
                    #print(vect)
            vects.append(vect)
        #print('VECTS ::::: ', vects)
        num = 0
        denum = 0

        sums = []

        for ind in range(len(vects[0])):

            for ind1 in range(self.nombre_edge):
                num += self.EDGES[ind1][4] * np.abs(vects[self.EDGES[ind1][1]][ind] - vects[self.EDGES[ind1][2]][ind])

            for h in range(self.nombre_demande):
                denum += self.DEMANDE[h][2] * np.abs(vects[int(self.DEMANDE[h][0])][ind] - vects[int(self.DEMANDE[h][1])][ind])
            if denum > 0:
                sums.append(num / denum)
            else:
                sums.append(np.infty)
        min_dim = np.argmin(sums)
        print('min dim :::', min_dim)
        vectsbis = [[vects[i][min_dim], i] for i in range(len(vects))]
        vectsort = sorted(vectsbis, key=lambda x: x[0])
        vectsortbis = [vectsort[i][1] for i in range(len(vectsort))]

        return vectsortbis, vects

    def find_sp_for_all_coordinates(self, random1):
        vects = self.compute_plongement(random1)[1]
        big_answer = []
        for coord in range(len(vects[0])):
            vectsbis = [[vects[i][coord], i] for i in range(len(vects))]
            vectsort = sorted(vectsbis, key=lambda x: x[0])
            vectsortbis = [vectsort[i][1] for i in range(len(vectsort))]

            nodes = vectsortbis
            sc = []
            for i in range(len(nodes) - 1):
                set = nodes[:i + 1]

                demandsum = 0
                capsum = 0
                for dem in self.DEMANDE:

                    if (dem[0] not in set and dem[1] in set) or (dem[0] in set and dem[1] not in set):

                        demandsum += dem[2]

                for edge in self.EDGES:
                    if (edge[1] not in set and edge[2] in set) or (edge[1] in set and edge[2] not in set):

                        capsum += edge[4]
                if demandsum > 0:
                    sc.append(capsum / demandsum)

                else:
                    sc.append(np.infty)
            temp = min(sc)
            res = []
            for idx in range(0, len(sc)):
                if temp == sc[idx]:
                    res.append(idx)
            big_answer.append([[nodes[:resi + 1] for resi in res], min(sc)])
        return big_answer

    def find_sp(self, random1):
        nodes = self.compute_plongement(random1)[0]

        sc = []
        for i in range(len(nodes) - 1):
            set = nodes[:i + 1]

            demandsum = 0
            capsum = 0
            for dem in self.DEMANDE:

                if (dem[0] not in set and dem[1] in set) or (dem[0] in set and dem[1] not in set):

                    demandsum += dem[2]

            for edge in self.EDGES:
                if (edge[1] not in set and edge[2] in set) or (edge[1] in set and edge[2] not in set):

                    capsum += edge[4]
            if demandsum > 0:
                sc.append(capsum / demandsum)

            else:
                sc.append(np.infty)
        print("CCCUUUTS :::: ", sc)

        temp = min(sc)
        res = []
        for idx in range(0, len(sc)):
            if temp == sc[idx]:
                res.append(idx)
        return [nodes[:resi + 1] for resi in res], min(sc)

g = MCF_graph_bis('../graphe normal/demande_short.csv', 'noeux_short.csv', 'edge_short.csv')
#g = MCF_graph_bis('Graphe biparti/graphs/Demands_bipartite(6, 6).csv', 'Graphe biparti/graphs/Nodes_bipartite(6, 6).csv', 'Graphe biparti/graphs/Edges_bipartite(6, 6).csv')
test_sp = []
normal_sp = g.find_sp(False)
print('normal :: ', normal_sp)

