import networkx as nx  # for generating and analyzing graphs               #import networkx for graph manipulations and analysis
import matplotlib.pyplot as plt  # for visualizations                      #import matplotlib for plotting and visualizing
import numpy as np  # for data manipulation                                 #import numpy for numerical computations
import random                                                             #import random for random selections
from collections import Counter                                            #import Counter to get distributions easily
import scipy.stats as stats                                               #import stats for fitting and analyzing distributions

# Define the parameters for the network models
num_nodes = 100                                                            #number of nodes in the graph
initial_degree = 3                                                         #initial edges per new node in Holme-Kim model
p_triangular_closure = 0.8                                                 #probability of triangle formation in Holme-Kim model

# Generate Holme-Kim model (scale-free with clustering)
holme_kim_graph = nx.powerlaw_cluster_graph(n=num_nodes,                   #generate Holme-Kim model graph 
                                            m=initial_degree, 
                                            p=p_triangular_closure)        #with a probability p for forming triangles

# Extract degree sequence from Holme-Kim graph for Configuration model
holme_kim_degree_sequence = [holme_kim_graph.degree(node)                  #extract degree sequence from the Holme-Kim graph
                             for node in holme_kim_graph.nodes]

# Generate Configuration model (based on Holme-Kim degree sequence)
configuration_graph = nx.configuration_model(holme_kim_degree_sequence)    #creates a multi-graph from the degree sequence

def Link_randomise_Graph(orig_net, num_rewirings):                         #function to randomize links by rewiring edges
    _copy_net = orig_net.copy()                                            #create a copy of the original network
    _rews = 0                                                              #counter for rewirings
    while _rews < num_rewirings:                                           #perform rewiring until desired number is reached
        _link_list = list(_copy_net.edges())                               #list of edges in the graph
        _rand_edge_inds = np.random.randint(0, len(_link_list), 2)         #randomly select two distinct edges
        if _rand_edge_inds[0] != _rand_edge_inds[1]:                       #ensure chosen edges are different
            _s1, _t1 = _link_list[_rand_edge_inds[0]][0], _link_list[_rand_edge_inds[0]][1]  #extract endpoints of first edge
            _s2, _t2 = _link_list[_rand_edge_inds[1]][0], _link_list[_rand_edge_inds[1]][1]  #extract endpoints of second edge
            if len(set([_s1, _t1, _s2, _t2])) == 4:                        #check edges don't share any node
                _s1_neighs = _copy_net.neighbors(_s1)                      #neighbors of s1
                _s2_neighs = _copy_net.neighbors(_s2)                      #neighbors of s2
                # Check no parallel edges formed by the rewiring
                if (not _t2 in _s1_neighs) and (not _t1 in _s2_neighs):
                    _copy_net.remove_edge(_s1, _t1)                        #remove first edge
                    _copy_net.remove_edge(_s2, _t2)                        #remove second edge
                    _copy_net.add_edge(_s1, _t2)                           #add rewired edge s1-t2
                    _copy_net.add_edge(_s2, _t1)                           #add rewired edge s2-t1
                    _rews += 1                                             #increment rewiring count
    return _copy_net                                                        #return the rewired network

def adjust_graph_without_weights(graph):
    """
    Modify a MultiGraph/MultiDiGraph to a simple Graph:
    1. Remove loops and parallel edges.
    2. Add new edges to maintain original edge count, ensuring no loops or parallels.
    """
    if not isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):            #check type
        raise TypeError("Input must be a MultiGraph or MultiDiGraph.")

    # Step 1: Create a simple graph (no loops or parallel edges)
    simple_graph = nx.Graph()                                              #simple undirected graph
    for u, v in graph.edges():                                             #iterate over original edges
        if u != v:                                                         #avoid self-loops
            simple_graph.add_edge(u, v)                                    #add edge to simple graph

    # Count removed edges
    original_edge_count = graph.number_of_edges()                          #count original edges
    simplified_edge_count = simple_graph.number_of_edges()                 #count simplified edges
    removed_edges = original_edge_count - simplified_edge_count            #how many edges were removed

    print(f"Removed {removed_edges} edges (loops or parallels)")            #inform user

    # Step 2: Add new edges to restore the original edge count
    nodes = list(simple_graph.nodes)                                       #list of nodes in the simplified graph
    added_edges = 0                                                        #count of added edges

    while added_edges < removed_edges:                                     #add edges until original count is restored
        u, v = random.sample(nodes, 2)                                     #random pick distinct nodes
        if not simple_graph.has_edge(u, v):                                #ensure no parallel edge
            simple_graph.add_edge(u, v)                                    #add the new edge
            added_edges += 1                                               #increment added edges

    print(f"Added {added_edges} new edges to maintain the original edge count.")  #inform user

    return simple_graph                                                    #return the adjusted simple graph

# Randomize the configuration graph and ensure it's simple
configuration_graph = Link_randomise_Graph(configuration_graph, 4000)      #perform 4000 rewirings
configuration_graph = adjust_graph_without_weights(configuration_graph)     #ensure final configuration_graph is simple

# ----------------------------------------------------------------------
# Compute metrics for a given graph
# ----------------------------------------------------------------------
def compute_metrics(G):
    metrics = {}
    metrics['clustering'] = nx.clustering(G)                               #Clustering Coeff
    metrics['knn'] = nx.average_neighbor_degree(G)                         #Average Nearest Neighbor Degree
    metrics['closeness'] = nx.closeness_centrality(G)                      #Closeness Centrality
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)      #Degree Assortativity
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
degree_dict_conf = dict(configuration_graph.degree())                     #get degrees in config graph
M_conf = configuration_graph.number_of_edges()                             #total edges in config graph
pair_connection_probabilities = {}                                        #store connection probabilities

for i in configuration_graph.nodes():                                      #iterate over pairs (i,j)
    for j in configuration_graph.nodes():
        if j > i:                                                          #avoid double counting
            k_i = degree_dict_conf[i]                                      #degree of node i
            k_j = degree_dict_conf[j]                                      #degree of node j
            p_ij = (k_i * k_j) / (2.0 * M_conf)                            #theoretical probability of i-j connection
            pair_connection_probabilities[(i, j)] = p_ij

# ----------------------------------------------------------------------
# Functions for various plots and analyses
# ----------------------------------------------------------------------
def plot_degree_distribution(G, graph_label):
    # Extract degree distribution
    degrees = [d for _, d in G.degree()]                                   #list of degrees
    degree_count = Counter(degrees)                                        #frequency of each degree
    deg, cnt = zip(*sorted(degree_count.items()))                          #sort by degree

    # Plot linear scale degree distribution
    plt.figure()
    plt.bar(deg, cnt, width=0.8, color='b', alpha=0.7)                     #bar plot for degree counts
    plt.xlabel("Degree")                                                   #x-axis label
    plt.ylabel("Count")                                                    #y-axis label
    plt.title(f"Degree Distribution ({graph_label})")                      #title with graph label
    plt.show()

    # Plot log-log scale degree distribution
    plt.figure()
    plt.loglog(deg, cnt, 'o', markersize=5, alpha=0.7)                     #log-log plot
    plt.xlabel("Degree (log)")                                             #x-axis label (log)
    plt.ylabel("Count (log)")                                              #y-axis label (log)
    plt.title(f"Degree Distribution - Log-Log ({graph_label})")            #log-log title
    plt.show()

def plot_clustering_distribution(G, graph_label):
    clustering_vals = list(nx.clustering(G).values())                      #list of clustering values
    plt.figure()
    plt.hist(clustering_vals, bins=30, alpha=0.7, color='g')               #histogram of clustering coeff
    plt.xlabel("Clustering Coefficient")                                   #x-axis label
    plt.ylabel("Frequency")                                                #y-axis label
    plt.title(f"Clustering Coefficient Distribution ({graph_label})")       #title with graph label
    plt.show()

def small_world_comparison(G, graph_label):
    n = len(G)                                                             #number of nodes
    k_approx = np.mean([d for _, d in G.degree()])                         #average degree
    p_equiv = k_approx / (n - 1)                                           #prob for Erdos-Renyi equivalent
    er_equiv = nx.erdos_renyi_graph(n, p_equiv)                            #generate random ER graph
    if nx.is_connected(G) and nx.is_connected(er_equiv):
        L = nx.average_shortest_path_length(G)                             #avg path length of original
        C = nx.average_clustering(G)                                       #avg clustering of original
        L_rand = nx.average_shortest_path_length(er_equiv)                 #avg path length of ER
        C_rand = nx.average_clustering(er_equiv)                           #avg clustering of ER
        print(f"\n--- Small World Comparison ({graph_label}) ---")
        print(f"Average Path Length (G): {L}")
        print(f"Clustering (G): {C}")
        print(f"Average Path Length (Random): {L_rand}")
        print(f"Clustering (Random): {C_rand}")
        print(f"Small-World Effect: (C/C_rand): {C/C_rand:.2f}, (L/L_rand): {L/L_rand:.2f}")
        return L, C, L_rand, C_rand
    else:
        print(f"\n{graph_label} graph or random equivalent not connected, unable to compute small-world metrics.")
        return None, None, None, None

def fit_power_law(G, graph_label):
    # Fit power-law by linear regression on log-log scale of degree distribution
    degrees = np.array([d for _, d in G.degree() if d > 0])                 #filter zero-degree nodes
    if len(degrees) < 2:
        print(f"Not enough degrees to fit power-law for {graph_label}")
        return None, None
    counts = Counter(degrees)                                              #count degrees
    deg, cnt = zip(*counts.items())                                        #get degrees and counts
    deg = np.array(deg)
    cnt = np.array(cnt, dtype=float)

    # Avoid issues with zero counts (should not happen) or log of zero
    if np.any(deg <= 0) or np.any(cnt <= 0):
        print(f"Cannot fit power-law for {graph_label} due to non-positive values.")
        return None, None

    x = np.log(deg)                                                        #log of degrees
    y = np.log(cnt)                                                        #log of counts
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)   #linear fit

    plt.figure()
    plt.loglog(deg, cnt, 'o', alpha=0.7, label='Data')                     #original data
    plt.loglog(deg, np.exp(intercept + slope*x), 'r--', label=f'Fit slope={slope:.2f}')
    plt.xlabel('Degree (log)')                                             #x label
    plt.ylabel('Count (log)')                                              #y label
    plt.title(f"Power-law fit ({graph_label})")                            #title
    plt.legend()
    plt.show()
    print(f"Power-law fit for {graph_label}: slope={slope}, R^2={r_value**2}")
    return slope, r_value**2

def print_sparseness_info(G, graph_label):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_degree = 2.0 * e / n
    print(f"\n--- Sparseness Info ({graph_label}) ---")
    print(f"Number of nodes: {n}")
    print(f"Number of edges: {e}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Graph is {'sparse' if e < 5*n else 'dense'} by rough comparison.")

def print_average_clustering_info(G, graph_label):
    c = nx.average_clustering(G)                                           #average clustering
    print(f"\n--- Clustering Info ({graph_label}) ---")
    print(f"Average Clustering Coefficient: {c:.4f}")

def get_node_metrics(G):
    """
    Extract node-level metrics (degree, clustering, knn) arrays for further correlation/analysis.
    """
    degrees = np.array([d for _, d in G.degree()])                          #degrees of all nodes
    nodes = list(G.nodes())                                                #list of nodes
    clustering_values = np.array([nx.clustering(G, n) for n in nodes])     #clustering value per node
    knn_dict = nx.average_neighbor_degree(G)                               #knn dictionary
    knn_values = np.array([knn_dict[n] for n in nodes])                    #knn values per node
    return degrees, clustering_values, knn_values

holme_kim_degrees, holme_kim_clustering_values, holme_kim_knn_values = get_node_metrics(holme_kim_graph)
config_degrees, config_clustering_values, config_knn_values = get_node_metrics(configuration_graph)

def plot_metric_vs_degree(deg_array, metric_array, graph_label, metric_name, loglog=False):
    """
    Plot a given metric as a function of degree.
    Can be linear or log-log scale.
    """
    plt.figure(figsize=(7,5))
    valid_idx = (deg_array > 0) & (metric_array > 0)  # Ensure positive values for log-log fits
    deg_filtered = deg_array[valid_idx]
    metric_filtered = metric_array[valid_idx]

    plt.scatter(deg_filtered, metric_filtered, alpha=0.7, edgecolor='k', label='Data')  #scatter plot of metric vs degree

    if loglog and len(deg_filtered) > 2 and np.all(deg_filtered > 0) and np.all(metric_filtered > 0):
        plt.xscale('log')                                                  #log scale on x
        plt.yscale('log')                                                  #log scale on y
        x_log = np.log(deg_filtered)
        y_log = np.log(metric_filtered)
        if len(x_log) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            fit_line = np.exp(intercept + slope * x_log)
            plt.plot(deg_filtered, fit_line, 'r--', label=f'Fit slope={slope:.2f}, R²={r_value**2:.2f}')

    plt.xlabel("Degree")                                                   #x-axis label
    plt.ylabel(metric_name)                                                #y-axis label
    title_str = f"{metric_name} vs Degree ({graph_label})"
    if loglog:
        title_str += " [Log-Log]"
    plt.title(title_str)                                                   #plot title
    plt.grid(True)
    plt.legend()
    plt.show()

# Clustering vs Degree
plot_metric_vs_degree(holme_kim_degrees, holme_kim_clustering_values, "Holme-Kim Graph", "Clustering Coefficient")
plot_metric_vs_degree(config_degrees, config_clustering_values, "Configuration Graph", "Clustering Coefficient")

# KNN vs Degree
plot_metric_vs_degree(holme_kim_degrees, holme_kim_knn_values, "Holme-Kim Graph", "Average Neighbor Degree (KNN)")
plot_metric_vs_degree(config_degrees, config_knn_values, "Configuration Graph", "Average Neighbor Degree (KNN)")

# Visualize Holme-Kim Graph
plt.figure(figsize=(8, 8))                                                 #figure for Holme-Kim visualization
nx.draw_spring(holme_kim_graph, node_size=10, node_color="blue", edge_color="gray", alpha=0.5)
plt.title("Holme-Kim Model Visualization")
plt.show()

# Visualize Configuration Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(configuration_graph, node_size=10, node_color="red", edge_color="gray", alpha=0.5)
plt.title("Configuration Model Visualization")
plt.show()

# Plot Degree Distributions
plot_degree_distribution(holme_kim_graph, "Holme-Kim Graph")
plot_degree_distribution(configuration_graph, "Configuration Graph")

# Plot Clustering Coefficient Distributions
plot_clustering_distribution(holme_kim_graph, "Holme-Kim Graph")
plot_clustering_distribution(configuration_graph, "Configuration Graph")

# Small-World Effect Comparison
hk_L, hk_C, hk_L_rand, hk_C_rand = small_world_comparison(holme_kim_graph, "Holme-Kim Graph")     #perform small-world comparison for Holme-Kim
conf_L, conf_C, conf_L_rand, conf_C_rand = small_world_comparison(configuration_graph, "Configuration Graph") #perform small-world comparison for Configuration graph

# Power-Law Fit
hk_slope, hk_r2 = fit_power_law(holme_kim_graph, "Holme-Kim Graph")        #fit power-law to Holme-Kim
conf_slope, conf_r2 = fit_power_law(configuration_graph, "Configuration Graph") #fit power-law to Configuration graph

# Sparseness and Average Degree
print_sparseness_info(holme_kim_graph, "Holme-Kim Graph")
print_sparseness_info(configuration_graph, "Configuration Graph")

# Average Clustering Info
print_average_clustering_info(holme_kim_graph, "Holme-Kim Graph")
print_average_clustering_info(configuration_graph, "Configuration Graph")

# Metrics Summary
print("\nMetrics Summary:")
print("Holme-Kim Graph Assortativity:", holme_kim_metrics['assortativity'])
print("Configuration Graph Assortativity:", config_metrics['assortativity'])

print("Holme-Kim Graph Avg Path Length:", holme_kim_metrics['avg_path_length'])
print("Configuration Graph Avg Path Length:", config_metrics['avg_path_length'])

# Additional plots for linear and log-log correlation analysis
plot_metric_vs_degree(holme_kim_degrees, holme_kim_clustering_values, "Holme-Kim Graph", "Clustering Coefficient", loglog=False)
plot_metric_vs_degree(holme_kim_degrees, holme_kim_clustering_values, "Holme-Kim Graph", "Clustering Coefficient", loglog=True)

plot_metric_vs_degree(config_degrees, config_clustering_values, "Configuration Graph", "Clustering Coefficient", loglog=False)
plot_metric_vs_degree(config_degrees, config_clustering_values, "Configuration Graph", "Clustering Coefficient", loglog=True)

plot_metric_vs_degree(holme_kim_degrees, holme_kim_knn_values, "Holme-Kim Graph", "Average Neighbor Degree (KNN)", loglog=False)
plot_metric_vs_degree(holme_kim_degrees, holme_kim_knn_values, "Holme-Kim Graph", "Average Neighbor Degree (KNN)", loglog=True)

plot_metric_vs_degree(config_degrees, config_knn_values, "Configuration Graph", "Average Neighbor Degree (KNN)", loglog=False)
plot_metric_vs_degree(config_degrees, config_knn_values, "Configuration Graph", "Average Neighbor Degree (KNN)", loglog=True)

# Final Graph Visualizations with updated node size and titles
plt.figure(figsize=(8,8))
nx.draw_spring(holme_kim_graph, node_size=20, node_color="blue", edge_color="gray", alpha=0.5)
plt.title("Holme-Kim Model - Final Visualization")
plt.show()

plt.figure(figsize=(8,8))
nx.draw_spring(configuration_graph, node_size=20, node_color="red", edge_color="gray", alpha=0.5)
plt.title("Configuration Model - Final Visualization")
plt.show()

# ---------------------------------------------
# Additional comparison plots based on the computed small-world and power-law values:
# First, check that we have valid values from the small-world comparison:
if hk_L is not None and hk_C is not None and hk_L_rand is not None and hk_C_rand is not None:
    hk_avg_path_g = hk_L
    hk_clustering_g = hk_C
    hk_avg_path_rand = hk_L_rand
    hk_clustering_rand = hk_C_rand
    hk_c_over_c_rand = hk_clustering_g / hk_clustering_rand if hk_clustering_rand != 0 else None
    hk_l_over_l_rand = hk_avg_path_g / hk_avg_path_rand if hk_avg_path_rand != 0 else None
else:
    hk_avg_path_g = hk_clustering_g = hk_avg_path_rand = hk_clustering_rand = hk_c_over_c_rand = hk_l_over_l_rand = None

if conf_L is not None and conf_C is not None and conf_L_rand is not None and conf_C_rand is not None:
    conf_avg_path_g = conf_L
    conf_clustering_g = conf_C
    conf_avg_path_rand = conf_L_rand
    conf_clustering_rand = conf_C_rand
    conf_c_over_c_rand = conf_clustering_g / conf_clustering_rand if conf_clustering_rand != 0 else None
    conf_l_over_l_rand = conf_avg_path_g / conf_avg_path_rand if conf_avg_path_rand != 0 else None
else:
    conf_avg_path_g = conf_clustering_g = conf_avg_path_rand = conf_clustering_rand = conf_c_over_c_rand = conf_l_over_l_rand = None

hk_power_slope = hk_slope
hk_power_r2 = hk_r2
conf_power_slope = conf_slope
conf_power_r2 = conf_r2

hk_assortativity = holme_kim_metrics['assortativity']
conf_assortativity = config_metrics['assortativity']

hk_avg_path_length = holme_kim_metrics['avg_path_length']
conf_avg_path_length = config_metrics['avg_path_length']

# Only plot if we have valid small-world data:
if hk_avg_path_g is not None and hk_avg_path_rand is not None and conf_avg_path_g is not None and conf_avg_path_rand is not None:
    # 1. Small-World Effect Comparison: Average Path Length
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(['G', 'Random'], [hk_avg_path_g, hk_avg_path_rand], color=['blue', 'gray'], alpha=0.7)
    axes[0].set_title("Holme-Kim Avg Path Length")
    axes[0].set_ylabel("Average Path Length")

    axes[1].bar(['G', 'Random'], [conf_avg_path_g, conf_avg_path_rand], color=['red', 'gray'], alpha=0.7)
    axes[1].set_title("Configuration Avg Path Length")

    plt.tight_layout()
    plt.show()

    # 2. Small-World Effect Comparison: Clustering
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(['G', 'Random'], [hk_clustering_g, hk_clustering_rand], color=['blue', 'gray'], alpha=0.7)
    axes[0].set_title("Holme-Kim Clustering")
    axes[0].set_ylabel("Clustering Coefficient")

    axes[1].bar(['G', 'Random'], [conf_clustering_g, conf_clustering_rand], color=['red', 'gray'], alpha=0.7)
    axes[1].set_title("Configuration Clustering")

    plt.tight_layout()
    plt.show()

    # 3. Small-World Ratios: C/C_rand and L/L_rand
    if hk_c_over_c_rand is not None and hk_l_over_l_rand is not None and conf_c_over_c_rand is not None and conf_l_over_l_rand is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].bar(['C/C_rand', 'L/L_rand'], [hk_c_over_c_rand, hk_l_over_l_rand], color=['blue', 'blue'], alpha=0.7)
        axes[0].set_title("Holme-Kim Small-World Effect")
        axes[0].set_ylabel("Ratio")

        axes[1].bar(['C/C_rand', 'L/L_rand'], [conf_c_over_c_rand, conf_l_over_l_rand], color=['red', 'red'], alpha=0.7)
        axes[1].set_title("Configuration Small-World Effect")

        plt.tight_layout()
        plt.show()

# Check if power-law fits are available before plotting
if hk_power_slope is not None and conf_power_slope is not None:
    # 4. Power-law Fit Slopes and R² Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(['Holme-Kim', 'Configuration'], [hk_power_slope, conf_power_slope], color=['blue', 'red'], alpha=0.7)
    ax1.set_title("Power-law Slopes")
    ax1.set_ylabel("Slope")

    ax2.bar(['Holme-Kim', 'Configuration'], [hk_power_r2, conf_power_r2], color=['blue', 'red'], alpha=0.7)
    ax2.set_title("Power-law R²")
    ax2.set_ylabel("R²")

    plt.tight_layout()
    plt.show()

# Check if assortativity and path lengths are available
if hk_assortativity is not None and conf_assortativity is not None and hk_avg_path_length is not None and conf_avg_path_length is not None:
    # 5. Assortativity and Average Path Length Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(['Holme-Kim', 'Configuration'], [hk_assortativity, conf_assortativity], color=['blue', 'red'], alpha=0.7)
    ax1.set_title("Assortativity Comparison")
    ax1.set_ylabel("Assortativity Coefficient")

    ax2.bar(['Holme-Kim', 'Configuration'], [hk_avg_path_length, conf_avg_path_length], color=['blue', 'red'], alpha=0.7)
    ax2.set_title("Average Path Length Comparison")
    ax2.set_ylabel("Average Path Length")

    plt.tight_layout()
    plt.show()
