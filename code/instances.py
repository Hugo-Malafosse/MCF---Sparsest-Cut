import csv
import os
import numpy as np
import pandas as pd
import networkx as nx


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


def create_complete_bipartite(left, right, repository_path=""):
    if repository_path != "" and not os.path.exists(repository_path):
        os.makedirs(repository_path)

    nodes_path = f"{repository_path}/Nodes_bipartite({left}, {right}).csv"
    edges_path = f"{repository_path}/Edges_bipartite({left}, {right}).csv"
    demands_path = f"{repository_path}/Demands_bipartite({left}, {right}).csv"

    Nodes = pd.DataFrame(columns=["node", "type", "site", "demand", "Y", "X", "server"])
    for k in range(left):
        Nodes.loc[k] = [k, 0, k + 1, 0, (k + 1) * 10, 10, 0]
    for k in range(right):
        Nodes.loc[left + k] = [left + k, 0, left + k + 1, 0, (k + 1) * 10, 20, 0]
    Nodes.to_csv(nodes_path, index=False, sep=";")

    Edges = pd.DataFrame(columns=["index", "nodeA", "nodeB", "cost", "capa", "type"])
    idx = 1
    for k in range(left):
        for p in range(right):
            Edges.loc[idx - 1] = [idx, k, left + p, 0, 1, 0]
            idx += 1
    Edges.to_csv(edges_path, index=False, sep=";")

    Demands = pd.DataFrame(columns=["nodeA", "nodeB", "demand"])
    if left != 1:
        for k in range(left):
            if k == left - 1 and k > 1:
                Demands.loc[k] = [k, 0, 1]
            else:
                Demands.loc[k] = [k, k + 1, 1]

    if right > 1:
        for k in range(right - 1):
            Demands.loc[left + k] = [left + k + 1, left + k, 1]

    Demands.to_csv(demands_path, index=False, sep=";")
    return nodes_path, edges_path, demands_path


def create_random_line(n, repository_path=""):
    if repository_path != "" and not os.path.exists(repository_path):
        os.makedirs(repository_path)

    nodes_path = f"{repository_path}/Nodes_random_line({n}).csv"
    edges_path = f"{repository_path}/Edges__random_line({n}).csv"
    demands_path = f"{repository_path}/Demands__random_line({n}).csv"

    Nodes = pd.DataFrame(columns=["node", "type", "site", "demand", "Y", "X", "server"])
    Edges = pd.DataFrame(columns=["index", "nodeA", "nodeB", "cost", "capa", "type"])
    Demands = pd.DataFrame(columns=["nodeA", "nodeB", "demand"])

    rand_cap = np.random.randint(0, n, n)
    for k in range(n):
        Nodes.loc[k] = [k, 0, k + 1, 0, 20, k * 5, 0]
        if k < n - 1:
            Edges.loc[k] = [k, k, k + 1, 0, rand_cap[k], 0]

    Demands.loc[0] = [0, n - 1, 1]

    Nodes.to_csv(nodes_path, index=False, sep=";")
    Edges.to_csv(edges_path, index=False, sep=";")
    Demands.to_csv(demands_path, index=False, sep=";")
    return nodes_path, edges_path, demands_path


def is_path(nodes_path, edges_path, demands_path):

    NODES, EDGES, DEMANDE, nombre_edge, nombre_noeuds, nombre_demande = setup(demands_path, nodes_path, edges_path)

    G1 = nx.Graph()
    for i in range(nombre_noeuds):
        G1.add_node(i, label=NODES[i][0], col='blue')

    for i in range(nombre_edge):
        G1.add_edge(EDGES[i][1], EDGES[i][2])

    for s, t, d in DEMANDE:
        #print(len(list(nx.all_simple_paths(G1, int(s), int(t)))))
        if len(list(nx.all_simple_paths(G1, int(s), int(t)))) == 0:
            return False

    return True


def create_random_bipartite(left, right,  repository_path, threshold=0.5):
    if repository_path != "" and not os.path.exists(repository_path):
        os.makedirs(repository_path)

    nodes_path = f"{repository_path}/Nodes_bipartite({left}, {right}).csv"
    edges_path = f"{repository_path}/Edges_bipartite({left}, {right}).csv"
    demands_path = f"{repository_path}/Demands_bipartite({left}, {right}).csv"

    Nodes = pd.DataFrame(columns=["node", "type", "site", "demand", "Y", "X", "server"])
    for k in range(left):
        Nodes.loc[k] = [k, 0, k+1, 0, (k+1)*10, 10, 0]
    for k in range(right):
        Nodes.loc[left+k] = [left+k, 0, left+k+1, 0, (k+1)*10, 20, 0]
    Nodes.to_csv(nodes_path, index=False, sep=";")

    Edges = pd.DataFrame(columns=["index", "nodeA", "nodeB", "cost", "capa", "type"])
    idx = 1
    for k in range(left):
        for p in range(right):
            if np.random.rand() > threshold:
                Edges.loc[idx-1] = [idx, k, left+p, 0, 1, 0]
                idx += 1
    Edges.to_csv(edges_path, index=False, sep=";")

    Demands = pd.DataFrame(columns=["nodeA", "nodeB", "demand"])
    if left != 1:
        for k in range(left):
            if k == left-1 and k > 1:
                Demands.loc[k] = [k, 0, 1]
            else:
                Demands.loc[k] = [k, k+1, 1]

    if right > 1:
        for k in range(right-1):
            Demands.loc[left+k] = [left+k+1, left+k, 1]

    Demands.to_csv(demands_path, index=False, sep=";")

    if not is_path(nodes_path, edges_path, demands_path):
        create_random_bipartite(left, right, threshold=threshold, repository_path=repository_path)

    return nodes_path, edges_path, demands_path

