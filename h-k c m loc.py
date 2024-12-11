import networkx as nx  # for generating and analyzing graphs               #import networkx for graph manipulations and analysis
import matplotlib.pyplot as plt  # for visualizations                      #import matplotlib for plotting and visualizing
import numpy as np  # for data manipulation                                 #import numpy for numerical computations
import random                                                             #import random for random selections

# Define the parameters for the network models
n_nodes = 100                                                             #number of nodes in the graph
initial_degree = 3                                                        #initial number of edges for new nodes in Holme-Kim model
p_triangular_closure = 0.8                                                #probability of triangle formation in Holme-Kim model

# Holme-Kim Model generation (scale-free with tunable clustering)
holme_kim_graph = nx.powerlaw_cluster_graph(n=n_nodes, 
                                            m=initial_degree, 
                                            p=p_triangular_closure)       #generate Holme-Kim model graph with n nodes, initial_degree edges per new node, and p for clustering

# Degree sequence generation for Configuration Model
degree_sequence = [holme_kim_graph.degree(node) for node in holme_kim_graph.nodes]   #extract degree sequence from Holme-Kim graph

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
# Additional Metrics and Comparisons
# ----------------------------------------------------------------------
# Clustering Coefficient
holme_kim_clustering = nx.clustering(holme_kim_graph)                      #compute clustering for Holme-Kim graph
configuration_clustering = nx.clustering(configuration_graph)              #compute clustering for Configuration graph

# Average Nearest Neighbors Degree (KNN)
holme_kim_knn = nx.average_neighbor_degree(holme_kim_graph)                #compute KNN for Holme-Kim graph
configuration_knn = nx.average_neighbor_degree(configuration_graph)        #compute KNN for Configuration graph

# Closeness Centrality
holme_kim_closeness = nx.closeness_centrality(holme_kim_graph)             #closeness centrality for Holme-Kim
configuration_closeness = nx.closeness_centrality(configuration_graph)     #closeness centrality for Configuration graph

# Additional metrics:
# 1. Degree Assortativity
holme_kim_assortativity = nx.degree_assortativity_coefficient(holme_kim_graph)            #degree assortativity for Holme-Kim
configuration_assortativity = nx.degree_assortativity_coefficient(configuration_graph)    #degree assortativity for Configuration

# 2. Average Shortest Path Length (if the graph is connected)
#   We check connectivity because if the graph is not connected, average shortest path is not defined globally.
if nx.is_connected(holme_kim_graph):
    holme_kim_avg_path_length = nx.average_shortest_path_length(holme_kim_graph)          #average shortest path length for Holme-Kim
else:
    holme_kim_avg_path_length = None                                                     #not defined if graph is disconnected

if nx.is_connected(configuration_graph):
    configuration_avg_path_length = nx.average_shortest_path_length(configuration_graph)  #average shortest path length for Configuration
else:
    configuration_avg_path_length = None                                                 #not defined if graph is disconnected

# 3. Betweenness Centrality distribution (sample computation - may be expensive for large graphs)
holme_kim_betweenness = nx.betweenness_centrality(holme_kim_graph)           #compute betweenness centrality for Holme-Kim
configuration_betweenness = nx.betweenness_centrality(configuration_graph)   #compute betweenness centrality for Configuration

# ----------------------------------------------------------------------
# Probability that a pair of nodes i and j are connected in the Configuration Model
# ----------------------------------------------------------------------
# For a Configuration Model with given degree sequence {k_i}, 
# the expected probability that two distinct nodes i and j are connected is approximately:
# P(i-j connected) ~ (k_i * k_j) / (2 * M), where M = total number of edges = sum(k_i)/2.

# We can compute this probability for each pair and then compare with the actual existence of edges:
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

# Note: The above theoretical probability does not necessarily equal the empirical frequency in a single realization,
# but it gives an expected probability. For large graphs, the configuration model approaches this expectation.

# ----------------------------------------------------------------------
# Print out some of the newly computed metrics
print("Additional Metrics:")
print(f"Holme-Kim Degree Assortativity: {holme_kim_assortativity}")          #print degree assortativity for Holme-Kim
print(f"Configuration Degree Assortativity: {configuration_assortativity}")  #print degree assortativity for Configuration

print(f"Holme-Kim Avg Shortest Path Length: {holme_kim_avg_path_length}")     #print avg path length for Holme-Kim (if connected)
print(f"Configuration Avg Shortest Path Length: {configuration_avg_path_length}")  #print avg path length for Configuration (if connected)

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

# Plot clustering coefficient distributions
plt.figure(figsize=(10, 5))                                                 #create figure for clustering distribution
plt.hist(holme_kim_clustering.values(), bins=50, alpha=0.5, label="Holme-Kim Model")       #plot Holme-Kim clustering histogram
plt.hist(configuration_clustering.values(), bins=50, alpha=0.5, label="Configuration Model")#plot Config clustering histogram
plt.xlabel("Clustering Coefficient")                                        #x-axis label
plt.ylabel("Frequency")                                                     #y-axis label
plt.legend()                                                                #add legend
plt.title("Clustering Coefficient Distribution")                            #title
plt.show()                                                                  #display the figure

# Plot KNN vs Degree
degree_sequence_hk, knn_values_hk = zip(*sorted(holme_kim_knn.items()))     #sort and unzip Holme-Kim KNN items
degree_sequence_conf, knn_values_conf = zip(*sorted(configuration_knn.items()))  #sort and unzip Configuration KNN items

plt.figure(figsize=(10, 5))                                                 #create figure for KNN plot
plt.plot(degree_sequence_hk, knn_values_hk, 'o-', label="Holme-Kim Model")  #plot Holme-Kim KNN vs degree
plt.plot(degree_sequence_conf, knn_values_conf, 'x-', label="Configuration Model") #plot Config KNN vs degree
plt.xlabel("Degree")                                                        #x-axis label
plt.ylabel("Average Nearest Neighbors Degree (KNN)")                        #y-axis label
plt.legend()                                                                #add legend
plt.title("Average Nearest Neighbors Degree vs Degree")                     #title
plt.show()                                                                  #display the figure

# Plot closeness centrality distributions
plt.figure(figsize=(10, 5))                                                 #create figure for closeness distribution
plt.hist(holme_kim_closeness.values(), bins=50, alpha=0.5, label="Holme-Kim Model")         #plot Holme-Kim closeness histogram
plt.hist(configuration_closeness.values(), bins=50, alpha=0.5, label="Configuration Model") #plot Config closeness histogram
plt.xlabel("Closeness Centrality")                                          #x-axis label
plt.ylabel("Frequency")                                                     #y-axis label
plt.legend()                                                                #add legend
plt.title("Closeness Centrality Distribution")                              #title
plt.show()                                                                  #display the figure

# Note on scale-free model with high clustering:
# The Holme-Kim model used here generates a scale-free network (degree distribution follows a power law)
# and the parameter p (p_triangular_closure) controls the probability of adding edges
# that create triangles, thereby controlling the clustering level.
# A high p (close to 1) leads to a scale-free network with higher clustering.
