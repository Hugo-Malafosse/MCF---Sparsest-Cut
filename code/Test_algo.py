from MCF_class import MCF_graph
import matplotlib.pyplot as plt
import instances
import pandas as pd


def test_une_instance(n, nodeleft, noderight, path_demande, path_noeud, path_edge, dir):
    g = MCF_graph(path_demande, path_noeud, path_edge)
    sc_values = []
    iterations = [i for i in range(n)]
    MCF = g.find_ze()[1]
    Mcf_liste = [MCF for _ in range(n)]
    value_base = g.find_sp(False)[1]
    f = [value_base for i in range(n)]
    vects_normal = g.compute_plongement(False)[1]
    distor = g.compute_distorsion(vects_normal)
    distortion_normal = [distor for i in range(n)]
    distortion_random = []
    for i in range(n):
        sc_values.append(g.find_sp(True)[1])
        vects_random = g.compute_plongement(True)[1]
        distor_random = g.compute_distorsion(vects_random)
        distortion_random.append(distor_random)

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
    for i in range(n):
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

            normal = {'nodes left': i, 'nodes right': j, 'MCF': MCF, 'sparsest cut': SP[0], 'sparsest cut value': SP[1], 'distortion': distor_norm}
            random = {'nodes left': i, 'nodes right': j, 'MCF': MCF, 'sparsest cut': SPrand[0], 'sparsest cut value': SPrand[1], 'distortion': distor_rand}
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

    for i in grosse_liste:
        graph_no.append(i)
        random = grosse_liste[i]['randomized']
        normal = grosse_liste[i]['optimal']
        MCF.append(normal['MCF'])
        sp_norm.append(normal['sparsest cut value'])
        sp_rand.append(random['sparsest cut value'])
        distor_norm.append(normal['distortion'])
        distor_random.append(random['distortion'])

    plt.plot(graph_no, sp_norm, 'r', label='algorithme normal')
    plt.plot(graph_no, sp_rand, 'b', label='algorithme randomisé')
    plt.plot(graph_no, MCF, 'g', label='MCF value')
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


create_data(6, 20, '../Graphe biparti/graphs', 'complete')