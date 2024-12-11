import networkx as nx  # for generating and analyzing graphs               #import networkx for graph manipulations and analysis
import matplotlib.pyplot as plt  # for visualizations                      #import matplotlib for plotting and visualizing
import numpy as np  # for data manipulation                                 #import numpy for numerical computations
import random                                                             #import random for random selections
from collections import Counter                                            #import Counter to get distributions easily
import scipy.stats as stats                                               #import stats for fitting and analyzing distributions

# Define the parameters for the network models
n_nodes = 100                                                              #number of nodes in the graph
initial_degree = 3                                                         #initial number of edges for new nodes in Holme-Kim model
p_triangular_closure = 0.8                                                 #probability of triangle formation in Holme-Kim model

# Holme-Kim Model generation (scale-free with tunable clustering)
holme_kim_graph = nx.powerlaw_cluster_graph(n=n_nodes,                     #generate Holme-Kim model graph with n nodes, initial_degree edges per new node, and p for clustering
                                            m=initial_degree, 
                                            p=p_triangular_closure)       

# Degree sequence generation for Configuration Model
degree_sequence = [holme_kim_graph.degree(node) for node in holme_kim_graph.nodes]  #extract degree sequence from Holme-Kim graph

# Configuration Model generation
configuration_graph = nx.configuration_model(degree_sequence)              #creates a multi-graph realization of the given degree sequence

def Link_randomise_Graph(orig_net,num_rewirings):                          #function to randomize links by rewiring edges
    _copy_net = orig_net.copy();                                           #create a copy of the original network
    _rews = int(0);                                                        #counter for rewirings
    while _rews < num_rewirings:                                           #perform rewiring until we reach the desired number
        _link_list = list(_copy_net.edges());                              #list of edges in the graph
        _rand_edge_inds = np.random.randint(0,len(_link_list),2);          #randomly choose two distinct edges
        if _rand_edge_inds[0] != _rand_edge_inds[1]:                       #ensure that the two chosen edges are different
            _s1,_t1 = _link_list[_rand_edge_inds[0]][0],_link_list[_rand_edge_inds[0]][1];   #extract endpoints of first chosen edge
            _s2,_t2 = _link_list[_rand_edge_inds[1]][0],_link_list[_rand_edge_inds[1]][1];   #extract endpoints of second chosen edge
            if len(set([_s1,_t1,_s2,_t2])) == 4:                           #ensure that the two edges share no common endpoints (4 distinct nodes)
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
            simple_graph.add_edge(u, v)                                    #add the edge (avoids parallel edges in a simple graph)

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

# Randomize the configuration graph
configuration_graph = Link_randomise_Graph(configuration_graph, 4000)      #perform 4000 rewirings on the configuration graph
configuration_graph = adjust_graph_without_weights(configuration_graph)     #ensure final configuration_graph is a simple graph

# ----------------------------------------------------------------------
# Compute metrics for given graph
# ----------------------------------------------------------------------
def compute_metrics(G):
    metrics = {}
    # Clustering Coefficient
    metrics['clustering'] = nx.clustering(G)

    # Average Nearest Neighbors Degree (KNN)
    metrics['knn'] = nx.average_neighbor_degree(G)

    # Closeness Centrality
    metrics['closeness'] = nx.closeness_centrality(G)

    # Degree Assortativity
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)

    # Average Shortest Path Length (if connected)
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        metrics['avg_path_length'] = None

    # Betweenness Centrality
    metrics['betweenness'] = nx.betweenness_centrality(G)

    return metrics

# Compute metrics for Holme-Kim and Configuration models
holme_kim_metrics = compute_metrics(holme_kim_graph)
config_metrics = compute_metrics(configuration_graph)

# Probability that a pair of nodes i and j are connected in the configuration model
degree_dict_conf = dict(configuration_graph.degree())                      #get degrees of all nodes in configuration graph
M_conf = configuration_graph.number_of_edges()                              #total number of edges in configuration graph
pair_connection_probabilities = {}                                         #dict to store probabilities for each pair (i,j)
for i in configuration_graph.nodes():                                       #iterate over each node i
    for j in configuration_graph.nodes():
        if j > i:                                                           #consider only j > i to avoid double counting and loops
            k_i = degree_dict_conf[i]                                       #degree of node i
            k_j = degree_dict_conf[j]                                       #degree of node j
            p_ij = (k_i * k_j) / (2.0 * M_conf)                             #theoretical probability of i-j connection in config model
            pair_connection_probabilities[(i,j)] = p_ij

# ----------------------------------------------------------------------
# Functions for additional plots and analysis
# ----------------------------------------------------------------------
def plot_degree_distribution(G, label):
    degrees = [d for n, d in G.degree()]
    degree_count = Counter(degrees)
    deg, cnt = zip(*sorted(degree_count.items()))
    
    # Plot in linear scale
    plt.figure()
    plt.bar(deg, cnt, width=0.8, color='b', alpha=0.7)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title(f"Degree Distribution ({label})")
    plt.show()
    
    # Plot in log-log scale for scale-free analysis
    plt.figure()
    plt.loglog(deg, cnt, 'o', markersize=5, alpha=0.7)
    plt.xlabel("Degree (log)")
    plt.ylabel("Count (log)")
    plt.title(f"Degree Distribution - Log-Log ({label})")
    plt.show()

def plot_clustering_distribution(G, label):
    clustering_vals = list(nx.clustering(G).values())
    plt.figure()
    plt.hist(clustering_vals, bins=30, alpha=0.7, color='g')
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.title(f"Clustering Coefficient Distribution ({label})")
    plt.show()

def small_world_comparison(G, label):
    n = len(G)
    k_approx = np.mean([d for _, d in G.degree()])      # average degree
    # Generate an equivalent Erdos-Renyi random graph for comparison:
    p_equiv = k_approx / (n-1)
    er_equiv = nx.erdos_renyi_graph(n, p_equiv)
    if nx.is_connected(G) and nx.is_connected(er_equiv):
        L = nx.average_shortest_path_length(G)
        C = nx.average_clustering(G)
        L_rand = nx.average_shortest_path_length(er_equiv)
        C_rand = nx.average_clustering(er_equiv)
        print(f"\n--- Small World Comparison ({label}) ---")
        print(f"Average Path Length (G): {L}")
        print(f"Clustering (G): {C}")
        print(f"Average Path Length (Random): {L_rand}")
        print(f"Clustering (Random): {C_rand}")
        print(f"Small-World Effect: (C/C_rand): {C/C_rand:.2f}, (L/L_rand): {L/L_rand:.2f}")
    else:
        print(f"\n{label} graph or random equivalent not connected, unable to compute small-world metrics.")

def fit_power_law(G, label):
    degrees = np.array([d for _, d in G.degree() if d > 0])
    if len(degrees) < 2:
        print(f"Not enough degrees to fit power-law for {label}")
        return
    counts = Counter(degrees)
    deg, cnt = zip(*counts.items())
    deg = np.array(deg)
    cnt = np.array(cnt, dtype=float)

    x = np.log(deg)
    y = np.log(cnt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    plt.figure()
    plt.loglog(deg, cnt, 'o', alpha=0.7, label='Data')
    plt.loglog(deg, np.exp(intercept + slope*x), 'r--', label=f'Fit slope={slope:.2f}')
    plt.xlabel('Degree (log)')
    plt.ylabel('Count (log)')
    plt.title(f"Power-law fit ({label})")
    plt.legend()
    plt.show()
    print(f"Power-law fit for {label}: slope={slope}, R^2={r_value**2}")

def print_sparseness_info(G, label):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_degree = 2.0 * e / n
    print(f"\n--- Sparseness Info ({label}) ---")
    print(f"Number of nodes: {n}")
    print(f"Number of edges: {e}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Graph is {'sparse' if e < 5*n else 'dense'} by rough comparison.")

def print_clustering_info(G, label):
    c = nx.average_clustering(G)
    print(f"\n--- Clustering Info ({label}) ---")
    print(f"Average Clustering Coefficient: {c:.4f}")

# ----------------------------------------------------------------------
# Visualize the Holme-Kim Graph
plt.figure(figsize=(8, 8))                                                  #create a figure for the Holme-Kim visualization
nx.draw_spring(holme_kim_graph, node_size=10, node_color="blue", edge_color="gray", alpha=0.5)
plt.title("Holme-Kim Model")
plt.show()

# Visualize the Configuration Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(configuration_graph, node_size=10, node_color="red", edge_color="gray", alpha=0.5)
plt.title("Configuration Model")
plt.show()

# Plot Degree Distributions
plot_degree_distribution(holme_kim_graph, "Holme-Kim")
plot_degree_distribution(configuration_graph, "Configuration")

# Plot Clustering Coefficient Distributions
plot_clustering_distribution(holme_kim_graph, "Holme-Kim")
plot_clustering_distribution(configuration_graph, "Configuration")

# Small-World Effect Comparison
small_world_comparison(holme_kim_graph, "Holme-Kim")
small_world_comparison(configuration_graph, "Configuration")

# Power-Law Fit (Scale-Free Behavior)
fit_power_law(holme_kim_graph, "Holme-Kim")
fit_power_law(configuration_graph, "Configuration")

# Sparseness and Average Degree
print_sparseness_info(holme_kim_graph, "Holme-Kim")
print_sparseness_info(configuration_graph, "Configuration")

# Large Local Clustering Coefficient
print_clustering_info(holme_kim_graph, "Holme-Kim")
print_clustering_info(configuration_graph, "Configuration")

print("\nMetrics Summary:")
print("Holme-Kim Assortativity:", holme_kim_metrics['assortativity'])
print("Configuration Assortativity:", config_metrics['assortativity'])

print("Holme-Kim Avg Path Length:", holme_kim_metrics['avg_path_length'])
print("Configuration Avg Path Length:", config_metrics['avg_path_length'])
