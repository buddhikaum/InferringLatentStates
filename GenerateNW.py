import networkx as nx
import sys

N =int( sys.argv[1])
G = nx.binomial_graph(N,0.2,directed=True)
nx.write_adjlist(G,'adj_temp.txt')
