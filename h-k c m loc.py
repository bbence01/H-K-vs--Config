import networkx as nx  # for generating and analyzing graphs
import matplotlib.pyplot as plt  # for visualizations
import numpy as np  # for data manipulation
import random


# Define the parameters for the network models
n_nodes = 100  # Number of nodes in the graph
initial_degree = 3  # Initial number of edges for new nodes in Holme-Kim model
p_triangular_closure = 0.8  # Probability of triangle formation in Holme-Kim model

# Degree sequence generation for Configuration Model
#degree_sequence = [initial_degree for _ in range(n_nodes)]  # Example uniform degree sequence


# Holme-Kim Model generation
holme_kim_graph = nx.powerlaw_cluster_graph(n=n_nodes, m=initial_degree, p=p_triangular_closure)

# Degree sequence generation for Configuration Model
degree_sequence = [d for n, d in holme_kim_graph.degree()]
degree_sequence = [holme_kim_graph.degree(node) for node in holme_kim_graph.nodes]


# Configuration Model generation
#configuration_graph = nx.configuration_model(degree_sequence)  # Generates a graph based on the degree sequence

# Convert the configuration model graph to a simple graph by removing parallel edges and self-loops
#configuration_graph = nx.Graph(configuration_graph)  # Converts to simple graph
#configuration_graph.remove_edges_from(nx.selfloop_edges(configuration_graph))  # Remove self-loops


# Generate an initial simple graph with the desired degree sequence
#configuration_graph = nx.havel_hakimi_graph(degree_sequence)
#configuration_graph = nx.random_degree_sequence_graph(degree_sequence, tries=10000)

# Randomize the graph using double-edge swaps
#num_swaps = 100 * configuration_graph.number_of_edges()  # Adjust the multiplier as needed
#configuration_graph = nx.double_edge_swap(configuration_graph, nswap=num_swaps, max_tries=num_swaps * 10)


def rewire_edges(G):
    """
    Rewires self-loops and parallel edges in the graph G.
    """
    # Make a copy of the graph to avoid modifying the original
    H = G.copy()

    # Rewire self-loops
    loops = list(nx.selfloop_edges(H))
    for u, v in loops:
        # Remove the self-loop
        H.remove_edge(u, v)
        # Find a node to connect to u
        possible_nodes = list(set(H.nodes()) - {u})
        new_v = random.choice(possible_nodes)
        while H.has_edge(u, new_v):
            new_v = random.choice(possible_nodes)
        H.add_edge(u, new_v)

    # Rewire parallel edges
    for u, v in list(H.edges()):
        num_edges = H.number_of_edges(u, v)
        while num_edges > 1:
            # Remove one of the parallel edges
            H.remove_edge(u, v)
            num_edges -= 1
            # Rewire the edge to two other nodes
            possible_u = list(set(H.nodes()) - {u, v})
            new_u = random.choice(possible_u)
            possible_v = list(set(H.nodes()) - {new_u, u, v})
            new_v = random.choice(possible_v)
            while H.has_edge(new_u, new_v):
                new_v = random.choice(possible_v)
            H.add_edge(new_u, new_v)
    return H


def Link_randomise_Graph(orig_net,num_rewirings):   # 2nd argument is the number of rewirings
    _copy_net = orig_net.copy();
    _rews = int(0);
    while _rews < num_rewirings:
        _link_list = list(_copy_net.edges());
        _rand_edge_inds = np.random.randint(0,len(_link_list),2);
        if _rand_edge_inds[0] != _rand_edge_inds[1]:                  
            _s1,_t1 = _link_list[_rand_edge_inds[0]][0],_link_list[_rand_edge_inds[0]][1];
            _s2,_t2 = _link_list[_rand_edge_inds[1]][0],_link_list[_rand_edge_inds[1]][1];
            if len(set([_s1,_t1,_s2,_t2])) == 4:         
                _s1_neighs = _copy_net.neighbors(_s1);
                _s2_neighs = _copy_net.neighbors(_s2);
                if (not _t2 in _s1_neighs) and (not _t1 in _s2_neighs):
                    _copy_net.remove_edge(_s1,_t1);
                    _copy_net.remove_edge(_s2,_t2);
                    _copy_net.add_edge(_s1,_t2);
                    _copy_net.add_edge(_s2,_t1);
                    _rews += 1;  
    return _copy_net;

def adjust_graph(graph):
    """
    Modify a graph by:
    1. Removing loops and parallel edges.
    2. Adding new edges to maintain the original edge count.
    """
    if not isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Input must be a MultiGraph or MultiDiGraph.")

    # Step 1: Create a simple graph (remove loops and parallel edges)
    simple_graph = nx.Graph()  # Undirected simple graph
    for u, v, data in graph.edges(data=True):
        if u != v:  # Skip self-loops
            if simple_graph.has_edge(u, v):
                # Combine parallel edges' weight or attributes if needed
                simple_graph[u][v]['weight'] += data.get('weight', 1)
            else:
                # Add the edge with its attributes
                simple_graph.add_edge(u, v, **data)

    # Count removed edges
    original_edge_count = graph.number_of_edges()
    simplified_edge_count = simple_graph.number_of_edges()
    removed_edges = original_edge_count - simplified_edge_count

    print(f"Removed {removed_edges} edges (loops or parallels)")

    # Step 2: Add new edges to maintain the original edge count
    nodes = list(simple_graph.nodes)
    added_edges = 0

    while added_edges < removed_edges:
        # Randomly select two distinct nodes
        u, v = random.sample(nodes, 2)
        if not simple_graph.has_edge(u, v):  # Ensure no parallel edge is created
            simple_graph.add_edge(u, v)
            added_edges += 1

    print(f"Added {added_edges} new edges to maintain the original edge count.")

    return simple_graph


def adjust_graph_without_weights(graph):
    """
    Modify a graph by:
    1. Removing loops and parallel edges.
    2. Adding new edges to maintain the original edge count, ensuring no loops or parallels.
    """
    if not isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Input must be a MultiGraph or MultiDiGraph.")

    # Step 1: Create a simple graph (remove loops and parallel edges)
    simple_graph = nx.Graph()  # Undirected simple graph
    for u, v in graph.edges():
        if u != v:  # Skip self-loops
            simple_graph.add_edge(u, v)

    # Count removed edges
    original_edge_count = graph.number_of_edges()
    simplified_edge_count = simple_graph.number_of_edges()
    removed_edges = original_edge_count - simplified_edge_count

    print(f"Removed {removed_edges} edges (loops or parallels)")

    # Step 2: Add new edges to maintain the original edge count
    nodes = list(simple_graph.nodes)
    added_edges = 0

    while added_edges < removed_edges:
        # Randomly select two distinct nodes
        u, v = random.sample(nodes, 2)
        if not simple_graph.has_edge(u, v):  # Ensure no parallel edge is created
            simple_graph.add_edge(u, v)
            added_edges += 1

    print(f"Added {added_edges} new edges to maintain the original edge count.")

    return simple_graph


configuration_graph = nx.configuration_model(degree_sequence)



configuration_graph = Link_randomise_Graph(configuration_graph, 4000)


#configuration_graph = rewire_edges(configuration_graph)

configuration_graph = adjust_graph_without_weights(configuration_graph)


#configuration_graph = nx.Graph(configuration_graph)  # Convert to simple Graph



# Clustering Coefficient
holme_kim_clustering = nx.clustering(holme_kim_graph)
configuration_clustering = nx.clustering(configuration_graph)

# Average Nearest Neighbors Degree (KNN)
holme_kim_knn = nx.average_neighbor_degree(holme_kim_graph)
configuration_knn = nx.average_neighbor_degree(configuration_graph)


# Closeness Centrality
holme_kim_closeness = nx.closeness_centrality(holme_kim_graph)
configuration_closeness = nx.closeness_centrality(configuration_graph)


# Visualize Holme-Kim Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(holme_kim_graph, node_size=10, node_color="blue", edge_color="gray", alpha=0.5)
plt.title("Holme-Kim Model")
plt.show()

# Visualize Configuration Graph
plt.figure(figsize=(8, 8))
nx.draw_spring(configuration_graph, node_size=10, node_color="red", edge_color="gray", alpha=0.5)
plt.title("Configuration Model")
plt.show()

# Plot clustering coefficient
plt.figure(figsize=(10, 5))
plt.hist(holme_kim_clustering.values(), bins=50, alpha=0.5, label="Holme-Kim Model")
plt.hist(configuration_clustering.values(), bins=50, alpha=0.5, label="Configuration Model")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.legend()
plt.title("Clustering Coefficient Distribution")
plt.show()

# Plot KNN vs Degree
degree_sequence_hk, knn_values_hk = zip(*sorted(holme_kim_knn.items()))
degree_sequence_conf, knn_values_conf = zip(*sorted(configuration_knn.items()))

plt.figure(figsize=(10, 5))
plt.plot(degree_sequence_hk, knn_values_hk, 'o-', label="Holme-Kim Model")
plt.plot(degree_sequence_conf, knn_values_conf, 'x-', label="Configuration Model")
plt.xlabel("Degree")
plt.ylabel("Average Nearest Neighbors Degree (KNN)")
plt.legend()
plt.title("Average Nearest Neighbors Degree vs Degree")
plt.show()


# Plot closeness centrality
plt.figure(figsize=(10, 5))
plt.hist(holme_kim_closeness.values(), bins=50, alpha=0.5, label="Holme-Kim Model")
plt.hist(configuration_closeness.values(), bins=50, alpha=0.5, label="Configuration Model")
plt.xlabel("Closeness Centrality")
plt.ylabel("Frequency")
plt.legend()
plt.title("Closeness Centrality Distribution")
plt.show()

