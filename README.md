# MCF---Sparsest-Cut
Approximation of Sparsest cut of a graph 

## Requirements 
classical python libraries (csv, math, plt, pandas...etc.)

pulp - a python librairy for linear optimization

NetworkX - a python library for graph manipulation

## Running the code

The code is pretty straightforward : The sparsest cut of a graph is computed by the class method find_sp or find_sp_for_all in the MCF_pulp_dual (or MCF_pulp, or MCF_pulp_raf) class. The method requires the path to the Demands, Nodes, and Edges csv (The format of these tables can be found in the examples given in the repo (Nodes, Demands, Edges). The MCF_pulp uses the "path" dual formulation of the MCF problem. The MCF_pulp_raf uses the rafinned version of the algorithm. The MCF_pulp_dual uses the "demand" dual formulation of the MCF (far less complex than the regular path formulation).

The instances.py file creates instances of bipartite graphs to test the code with. The test_algo.py tests the algorithms with many methods, and exports figures of the results.  
