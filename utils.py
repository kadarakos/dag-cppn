import math
import secrets
import networkx as nx

#TODO numerically unstable
def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sine(x):
    return math.sin(x)

def cosine(x):
    return math.cos(x)

def square(x):
    return x ** 2

def tanh(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1) 

def abstanh(x):
    return abs(tanh(x))

def relu(x):
    return max(0, x)

def soft_plus(x):
    return math.log(1.0 + math.exp(x))

def gen_name(prefix=''):
    return prefix + secrets.token_urlsafe(6)
    

def build_networkx_graph(network):
    """
    Build networkx Graph object from Network.
    Linear in the number of edges O(|E|).
    """
    G = nx.Graph()
    for idx, e in network.edges.items():
        parent = e.parent.idx
        child = e.child.idx
        weight = e.weight
        # networkx takes care of duplicate nodes
        G.add_node(parent)
        G.add_node(child)
        G.add_edge(parent, child, weight)
    return G

def plot_network(network):
    """
    Build networkx graph from Network object and use the built in plotting.
    https://plot.ly/python/network-graphs/
    https://community.plot.ly/t/in-a-network-graph-how-do-i-highlight-the-network-components-when-hovering-over-them/23562
    https://medium.com/@anand0427/network-graph-with-at-t-data-using-plotly-a319f9898a02
    """
    G = build_networkx_graph(network)
    # Assign position to each node for plotting
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for n, p in pos.items():
        G.node[n]['pos'] = p

activation_functions = [sigmoid, tanh, sine, square, cosine, identity]
