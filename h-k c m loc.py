import networkx as nx  # for generating and analyzing graphs               #import networkx for graph manipulations and analysis
import matplotlib.pyplot as plt  # for visualizations                      #import matplotlib for plotting and visualizing
import numpy as np  # for data manipulation                                 #import numpy for numerical computations
import random                                                             #import random for random selections

# Define the parameters for the network models
n_nodes = 100                                                             #number of nodes in the graph
initial_degree = 3                                                        #initial number of edges for new nodes in Holme-Kim model
p_triangular_closure = 0.8                                                #probability of triangle formation in Holme-Kim model

# Holme-Kim Model generation (scale-free with tunable clustering)
holme_kim_graph = nx.powerlaw_cluster_graph(n=n_nodes,                    #generate Holme-Kim model graph with n nodes, initial_degree edges per new node, and p for clustering
                                            m=initial_degree, 
                                            p=p_triangular_closure)       

# Degree sequence generation for Configuration Model
degree_sequence = [holme_kim_graph.degree(node) for node in holme_kim_graph.nodes]  #extract degree sequence from Holme-Kim graph

# Configuration Model generation
configuration_graph = nx.configuration_model(degree_sequence)             #creates a multi-graph realization of the given degree sequence


def Link_randomise_Graph(orig_net,num_rewirings):                          #function to randomize links by rewiring edges
    _copy_net = orig_net.copy();                                           #create a copy of the original network
    _rews = int(0);                                                        #counter for rewirings
    while _rews < num_rewirings:                                           #perform rewiring until we reach the desired number
        _link_list = list(_copy_net.edges());                              #list of edges in the graph
        _rand_edge_inds = np.random.randint(0,len(_link_list),2);          #randomly choose two distinct edges
        if _rand_edge_inds[0] != _rand_edge_inds[1]:                       #ensure that the two chosen edges are different
            _s1,_t1 = _link_list[_rand_edge_inds[0]][0],_link_list[_rand_edge_inds[0]][1];   #extract endpoints of first chosen edge
            _s2,_t2 = _link_list[_rand_edge_inds[1]][0],_link_list[_rand_edge_inds[1]][1];   #extract endpoints of second chosen edge
            if len(set([_s1,_t1,_s2,_t2])) == 4:                           #ensure that the two edges share no common endpoints (simple 4 distinct nodes)
                _s1_neighs = _copy_net.neighbors(_s1);                     #neighbors of s1
                _s2_neighs = _copy_net.neighbors(_s2);                     #neighbors of s2
                if (not _t2 in _s1_neighs) and (not _t1 in _s2_neighs):    #check that rewiring does not create parallel edges
                    _copy_net.remove_edge(_s1,_t1);                        #remove the first original edge
                    _copy_net.remove_edge(_s2,_t2);                        #remove the second original edge
                    _copy_net.add_edge(_s1,_t2);                           #add a new edge between s1 and t2
                    _copy_net.add_edge(_s2,_t1);                           #add a new edge between s2 and t1
                    _rews += 1;                                            #increment rewiring count
    return _copy_net;                                                      #return the rewired network

def adjust_graph_without_weights(graph):
    """
    Modify a graph by:
    1. Removing loops and parallel edges.
    2. Adding new edges to maintain the original edge count, ensuring no loops or parallels.
    """
    if not isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):            #check that input is a multi-graph or multi-di-graph
        raise TypeError("Input must be a MultiGraph or MultiDiGraph.")

    # Step 1: Create a simple graph (remove loops and parallel edges)
    simple_graph = nx.Graph()                                              #initialize a simple undirected graph
    for u, v in graph.edges():                                             #iterate over edges in the original multi-graph
        if u != v:                                                         #skip self-loops
            simple_graph.add_edge(u, v)                                    #add the edge (automatically avoids parallel edges in a simple graph)

    # Count removed edges
    original_edge_count = graph.number_of_edges()                          #count edges in the original graph
    simplified_edge_count = simple_graph.number_of_edges()                 #count edges in the simplified graph
    removed_edges = original_edge_count - simplified_edge_count            #calculate how many edges were removed (loops/parallels)

    print(f"Removed {removed_edges} edges (loops or parallels)")            #inform how many edges were removed

    # Step 2: Add new edges to maintain the original edge count
    nodes = list(simple_graph.nodes)                                       #list of nodes in the simplified graph
    added_edges = 0                                                        #counter for added edges

    while added_edges < removed_edges:                                     #add edges until we restore the original edge count
        u, v = random.sample(nodes, 2)                                     #randomly choose two distinct nodes
        if not simple_graph.has_edge(u, v):                                #ensure no parallel edge is created
            simple_graph.add_edge(u, v)                                    #add the new edge
            added_edges += 1                                               #increment the number of added edges

    print(f"Added {added_edges} new edges to maintain the original edge count.")  #inform how many edges were added

    return simple_graph                                                    #return the adjusted simple graph

# Randomize the configuration graph using the defined rewiring function
configuration_graph = Link_randomise_Graph(configuration_graph, 4000)      #perform 4000 rewirings on the configuration graph

# Adjust the configuration graph by removing loops/parallels and adding new edges if needed
configuration_graph = adjust_graph_without_weights(configuration_graph)     #ensure final configuration_graph is a simple graph

# ----------------------------------------------------------------------
# ADDITIONAL TYPICAL GRAPHS FOR COMPARISON
# ----------------------------------------------------------------------

# 1. Erdos-Renyi (G(n, p)) Random Graph
p_er = 0.05                                                                #probability of edge existence between any pair of nodes
er_graph = nx.erdos_renyi_graph(n=n_nodes, p=p_er)                         #generate ER graph with n_nodes and edge probability p_er

# 2. Watts-Strogatz (Small-World) Graph
k_ws = 4                                                                   #each node is connected to k_ws nearest neighbors in a ring
p_ws = 0.1                                                                 #rewiring probability
ws_graph = nx.watts_strogatz_graph(n=n_nodes, k=k_ws, p=p_ws)              #generate Watts-Strogatz small-world graph

# 3. Barabasi-Albert (Scale-Free) Graph
# (Holme-Kim is a variation of scale-free. Here we also add classic Barabasi-Albert for comparison)
m_ba = 3                                                                   #edges to attach from a new node to existing nodes
ba_graph = nx.barabasi_albert_graph(n=n_nodes, m=m_ba)                     #generate BA scale-free graph

# 4. Random Geometric Graph
radius = 0.2                                                               #radius for connection in geometric space
rgg_graph = nx.random_geometric_graph(n=n_nodes, radius=radius)            #generate RGG

# ----------------------------------------------------------------------
# Compute metrics for all graphs (Holme-Kim, Configuration, ER, WS, BA, RGG)
# ----------------------------------------------------------------------

def compute_metrics(G):
    """
    Compute a set of metrics for a given graph G.
    Returns a dictionary of metrics.
    """
    metrics = {}
    # Clustering Coefficient
    metrics['clustering'] = nx.clustering(G)                                #compute clustering for G

    # Average Nearest Neighbors Degree (KNN)
    metrics['knn'] = nx.average_neighbor_degree(G)                          #compute KNN for G

    # Closeness Centrality
    metrics['closeness'] = nx.closeness_centrality(G)                       #closeness centrality for G

    # Degree Assortativity
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)       #degree assortativity for G

    # Average Shortest Path Length (if connected)
    if nx.is_connected(G):                                                 #check if G is connected
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)     #avg shortest path length if connected
    else:
        metrics['avg_path_length'] = None                                   #not defined if disconnected

    # Betweenness Centrality (sample metric - can be expensive for large graphs)
    metrics['betweenness'] = nx.betweenness_centrality(G)                   #betweenness centrality for G

    return metrics

# Compute metrics for each graph
holme_kim_metrics = compute_metrics(holme_kim_graph)
config_metrics = compute_metrics(configuration_graph)
er_metrics = compute_metrics(er_graph)
ws_metrics = compute_metrics(ws_graph)
ba_metrics = compute_metrics(ba_graph)
rgg_metrics = compute_metrics(rgg_graph)

# Print out some key metrics for comparison
print("\n--- KEY METRICS COMPARISON ---")
print("Holme-Kim Assortativity:", holme_kim_metrics['assortativity'])
print("Configuration Assortativity:", config_metrics['assortativity'])
print("ER Assortativity:", er_metrics['assortativity'])
print("WS Assortativity:", ws_metrics['assortativity'])
print("BA Assortativity:", ba_metrics['assortativity'])
print("RGG Assortativity:", rgg_metrics['assortativity'])

print("Holme-Kim Avg Path Length:", holme_kim_metrics['avg_path_length'])
print("Configuration Avg Path Length:", config_metrics['avg_path_length'])
print("ER Avg Path Length:", er_metrics['avg_path_length'])
print("WS Avg Path Length:", ws_metrics['avg_path_length'])
print("BA Avg Path Length:", ba_metrics['avg_path_length'])
print("RGG Avg Path Length:", rgg_metrics['avg_path_length'])

# ----------------------------------------------------------------------
# Probability that a pair of nodes i and j are connected in the Configuration Model
# (Already computed previously)
degree_dict_conf = dict(configuration_graph.degree())                      #get degrees of all nodes in configuration graph
M_conf = configuration_graph.number_of_edges()                              #total number of edges in configuration graph
pair_connection_probabilities = {}                                         #dict to store probabilities for each pair (i,j)

for i in configuration_graph.nodes():                                       #iterate over each node i
    for j in configuration_graph.nodes():
        if j > i:                                                           #consider only j > i to avoid double counting and loops
            k_i = degree_dict_conf[i]                                       #degree of node i
            k_j = degree_dict_conf[j]                                       #degree of node j
            p_ij = (k_i * k_j) / (2.0 * M_conf)                              #theoretical probability of i-j connection in config model
            pair_connection_probabilities[(i,j)] = p_ij                      #store the probability

# Visualize Holme-Kim Graph
plt.figure(figsize=(8, 8))                                                  #create a figure for the Holme-Kim visualization
nx.draw_spring(holme_kim_graph, node_size=10, node_color="blue", edge_color="gray", alpha=0.5)  #draw Holme-Kim graph
plt.title("Holme-Kim Model")                                                #set plot title
plt.show()                                                                  #display the figure

# Visualize Configuration Graph
plt.figure(figsize=(8, 8))                                                  #create a figure for the Configuration visualization
nx.draw_spring(configuration_graph, node_size=10, node_color="red", edge_color="gray", alpha=0.5) #draw Configuration graph
plt.title("Configuration Model")                                            #set plot title
plt.show()                                                                  #display the figure

# Visualize ER Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(er_graph, node_size=10, node_color="green", edge_color="gray", alpha=0.5)
plt.title("Erdos-Renyi (G(n,p)) Model")
plt.show()

# Visualize WS Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(ws_graph, node_size=10, node_color="orange", edge_color="gray", alpha=0.5)
plt.title("Watts-Strogatz Small-World Model")
plt.show()

# Visualize BA Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(ba_graph, node_size=10, node_color="purple", edge_color="gray", alpha=0.5)
plt.title("Barabasi-Albert Scale-Free Model")
plt.show()

# Visualize RGG
plt.figure(figsize=(8, 8))
# For RGG, we have node positions as an attribute: pos[node] = (x, y)
pos = nx.get_node_attributes(rgg_graph, 'pos')                              #get node positions for the RGG
nx.draw(rgg_graph, pos, node_size=10, node_color="cyan", edge_color="gray", alpha=0.5)
plt.title("Random Geometric Graph")
plt.show()

# Plot clustering coefficient distributions for all graphs
plt.figure(figsize=(10, 5))
plt.hist(holme_kim_metrics['clustering'].values(), bins=30, alpha=0.5, label="Holme-Kim")
plt.hist(config_metrics['clustering'].values(), bins=30, alpha=0.5, label="Configuration")
plt.hist(er_metrics['clustering'].values(), bins=30, alpha=0.5, label="Erdos-Renyi")
plt.hist(ws_metrics['clustering'].values(), bins=30, alpha=0.5, label="Watts-Strogatz")
plt.hist(ba_metrics['clustering'].values(), bins=30, alpha=0.5, label="Barabasi-Albert")
plt.hist(rgg_metrics['clustering'].values(), bins=30, alpha=0.5, label="RGG")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.legend()
plt.title("Clustering Coefficient Distribution for Various Graph Models")
plt.show()

# Note on scale-free models and high clustering:
# The Holme-Kim model is a variation of scale-free networks that also includes a mechanism to create triangles,
# resulting in higher clustering compared to the classic Barabasi-Albert scale-free model.
# As we can see from the metrics and distributions, different models have distinct structural properties.
