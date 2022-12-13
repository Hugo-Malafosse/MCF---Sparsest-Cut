from MCF_class import MCF_graph
import matplotlib.pyplot as plt
import instances
import pandas as pd
import numpy as np
from time import time
from MCF_class_dual import MCF_graph_bis

def test_une_instance(n, nodeleft, noderight, path_demande, path_noeud, path_edge, dir):
    g = MCF_graph(path_demande, path_noeud, path_edge)
    tv = g.brute_force()[1]
    truevalues = [tv for _ in range(n)]
    sc_values = []
    iterations = [i for i in range(n)]
    MCF = g.find_ze()[1]
    Mcf_liste = [MCF for _ in range(n)]
    value_base = g.find_sp(False)[1]
    f = [value_base for _ in range(n)]
    vects_normal = g.compute_plongement(False)[1]
    distor = g.compute_distorsion(vects_normal)
    distortion_normal = [distor for i in range(n)]
    distortion_random = []
    for _ in range(n):
        sc_values.append(g.find_sp(True)[1])
        vects_random = g.compute_plongement(True)[1]
        distor_random = g.compute_distorsion(vects_random)
        distortion_random.append(distor_random)

    plt.plot(iterations, truevalues, 'p', label='Valeur réelle de la sparsest cut')
    plt.plot(iterations, sc_values, 'r', label='algorithme randomisé')
    plt.plot(iterations, f, 'b', label='algorithme normal')
    plt.plot(iterations, Mcf_liste, 'g', label='MCF')
    plt.legend(loc="upper left")
    plt.xlabel('iterations')
    plt.ylabel('SP')
    plt.savefig(dir + '/sur un graphe bipartite avec ' + str(nodeleft) + ' noeuds à gauche et ' + str(noderight) + ' noeuds à droite' + '.png')
    plt.clf()
    plt.plot(iterations, distortion_normal, 'g', label='distortion normale')
    plt.plot(iterations, distortion_random, 'b', label='distortion random')
    plt.legend(loc="upper left")
    plt.xlabel('iterations')
    plt.ylabel('Distorsions')
    plt.savefig(dir + '/distortion sur un graphe bipartite avec ' + str(nodeleft) + ' noeuds à gauche et ' + str(noderight) + ' noeuds à droite' + '.png')
    plt.clf()


def get_exemple(itermax, path_demande, path_noeud, path_edge):

    for iter in range(itermax):

        test_une_instance(100, iter, path_demande, path_noeud, path_edge)


def repartition_sp(n, path_demande, path_noeud, path_edge):
    g = MCF_graph(path_demande, path_noeud, path_edge)
    sc_repartition = {}
    for _ in range(n):
        sp = str(g.find_sp(True)[0])
        if sp in sc_repartition:
            sc_repartition[sp] += 1
        else:
            sc_repartition[sp] = 1

    return sc_repartition, g.find_sp(False)[0]


def test_on_bipartite(n_bi, graph_type):
    Grosse_liste = {}

    k = 1
    for i in range(2, n_bi):
        for j in range(2, n_bi):
            print(k/((n_bi-1)**2))
            if graph_type == 'random':

                nodes_path, edges_path, demands_path = instances.create_random_bipartite(i, j, 'Graphe biparti/graphs')

            else:
                nodes_path, edges_path, demands_path = instances.create_complete_bipartite(i, j, 'Graphe biparti/graphs')

            g1 = MCF_graph(demands_path, nodes_path, edges_path)
            MCF = g1.find_ze()[1]
            SP = g1.find_sp(False)
            SPrand = g1.find_sp(True)
            vects_normal = g1.compute_plongement(False)[1]
            distor_norm = g1.compute_distorsion(vects_normal)
            vects_rand = g1.compute_plongement(True)[1]
            distor_rand = g1.compute_distorsion(vects_rand)
            bf = g1.brute_force()[1]
            normal = {'nodes left': i, 'nodes right': j, 'MCF': MCF, 'sparsest cut': SP[0], 'sparsest cut value': SP[1], 'distortion': distor_norm, 'Brute_force': bf}
            random = {'nodes left': i, 'nodes right': j, 'MCF': MCF, 'sparsest cut': SPrand[0], 'sparsest cut value': SPrand[1], 'distortion': distor_rand, 'Brute_force': bf}
            Grosse_liste[k] = {'optimal': normal, 'randomized': random}
            k += 1

    return Grosse_liste


def graph_all(grosse_liste):
    graph_no = []
    MCF = []
    sp_rand = []
    sp_norm = []
    distor_norm = []
    distor_random = []
    Brute_force = []
    diff_normal = []
    diff_random = []
    for i in grosse_liste:
        graph_no.append(i)
        random = grosse_liste[i]['randomized']
        normal = grosse_liste[i]['optimal']
        MCF.append(normal['MCF'])
        sp_norm.append(normal['sparsest cut value'])
        sp_rand.append(random['sparsest cut value'])
        distor_norm.append(normal['distortion'])
        distor_random.append(random['distortion'])
        Brute_force.append(normal['Brute_force'])
        diff_normal.append(np.abs(normal['Brute_force'] - normal['sparsest cut value'])/normal['Brute_force'])
        diff_random.append(np.abs(normal['Brute_force'] - random['sparsest cut value'])/normal['Brute_force'])

    plt.plot(graph_no, sp_norm, 'r', label='algorithme normal')
    plt.plot(graph_no, sp_rand, 'b', label='algorithme randomisé')
    plt.plot(graph_no, MCF, 'g', label='MCF value')
    plt.plot(graph_no, Brute_force, 'p', label='Brute Force value')
    plt.legend(loc="upper left")
    plt.xlabel('graph number')
    plt.ylabel('SP value and MCF')
    plt.savefig('./Graphe biparti/random vs opti on all graphs.png')
    plt.clf()
    plt.plot(graph_no, distor_random, 'b', label='distortion random')
    plt.plot(graph_no, distor_norm, 'g', label='distortion normale')
    plt.legend(loc="upper left")
    plt.xlabel('graph number')
    plt.ylabel('distortion')
    plt.savefig('./Graphe biparti/distortion random vs opti on all graphs.png')
    plt.clf()
    plt.plot(graph_no, diff_random, 'b', label='relative error on the randomized sparsest cut')
    plt.plot(graph_no, diff_normal, 'r', label='relative error on the normal sparsest cut')
    plt.legend(loc="upper left")
    plt.xlabel('graph number')
    plt.ylabel('relative error on the approximation')
    plt.savefig('./Graphe biparti/relative error on approx.png')
    plt.clf()

def create_data(n_graph, n_iter, repository_path, graph_type):

    grosse_liste = test_on_bipartite(n_graph, graph_type)

    big_csv = pd.DataFrame(columns=["nombre noeud gauche", "nombre noeud droit", "MCF", "sparsest cut opti", "sparsest cut value opti", "sparsest cut random", "sparsest cut value random"])

    for i in grosse_liste:
        random = grosse_liste[i]['randomized']
        normal = grosse_liste[i]['optimal']
        big_csv.loc[i] = [random['nodes left'], random['nodes right'], random['MCF'], normal['sparsest cut'], normal['sparsest cut value'], random['sparsest cut'], random['sparsest cut value']]
        left = random['nodes left']
        right = random['nodes right']
        nodes_path = f"{repository_path}/Nodes_bipartite({left}, {right}).csv"
        edges_path = f"{repository_path}/Edges_bipartite({left}, {right}).csv"
        demands_path = f"{repository_path}/Demands_bipartite({left}, {right}).csv"
        g1 = MCF_graph(demands_path, nodes_path, edges_path)
        g1.export_graph([1 for _ in range(g1.nombre_edge)], [], left, right)
        test_une_instance(n_iter, left, right, demands_path, nodes_path, edges_path,
                          '../Graphe biparti/sparsest cut random vs opti for every graph')

    big_csv.to_csv('Graphe biparti/big_csv.csv', index=False, sep=";")

    graph_all(grosse_liste)


def test_time(n_bi):
    nombre_node = {}
    time_brute = []
    time_normal = []
    time_random = []
    nodes = []
    for i in range(2, n_bi):
        for j in range(2, n_bi):
            
            nodes_path, edges_path, demands_path = instances.create_random_bipartite(i, j, 'Graphe biparti/graphs')
            g1 = MCF_graph_bis(demands_path, nodes_path, edges_path)

            t0 = time()
            p_random = g1.find_sp(True)
            t1 = time()
            t_brute = t1-t0
            sp_approx = g1.find_sp(False)
            t2 = time()
            t_normal = t2-t1
            sp_true = g1.brute_force()
            t3 = time()
            t_random = t3-t2

            if g1.nombre_noeuds in nombre_node:
                nombre_node[g1.nombre_noeuds]['normal'].append(t_normal)
                nombre_node[g1.nombre_noeuds]['random'].append(t_random)
                nombre_node[g1.nombre_noeuds]['brute'].append(t_brute)
            else:
                nombre_node[g1.nombre_noeuds] = {'normal': [t_normal], 'random': [t_random], 'brute': [t_brute]}
            
    for nodenum in nombre_node:
        nodes.append(nodenum)

        time_normal.append(np.sum(nombre_node[nodenum]['normal'])/len(nombre_node[nodenum]['normal']))
        time_random.append(np.mean(nombre_node[nodenum]['random'])/len(nombre_node[nodenum]['random']))
        time_brute.append(np.mean(nombre_node[nodenum]['brute'])/len(nombre_node[nodenum]['brute']))

    print('nodenum::   ', len(nodes))
    print('time norm :: ', len(time_normal))
    print(len(time_brute))
    print(len(time_random))
    plt.plot(nodes, time_normal, 'r', label='Time for normal algo')
    plt.plot(nodes, time_random, 'b', label='Time for randomized algo')
    plt.plot(nodes, time_brute, 'g', label='Time for brute force algo')
    plt.legend(loc="upper left")
    plt.xlabel('number of node in the graph')
    plt.ylabel('time taken by the algorithms')
    plt.savefig('./Graphe biparti/time complexity of different algorithms.png')
    plt.clf()


#create_data(6, 20, '../Graphe biparti/graphs', 'complete')
# grosse_liste = test_on_bipartite(6, 'random')
# graph_all(grosse_liste)
test_time(8)
