import math
import secrets
import networkx as nx
import plotly.graph_objects as go


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
        G.add_edge(parent, child)
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
        G.nodes[n]['pos'] = p
    
    edge_trace = go.Scatter(x=[], y=[],
                            line=dict(width=0.5,color='#888'),
                            hoverinfo='none',
                            mode='lines')
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(x=[], y=[],
                            text=[],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(showscale=True,
                                        colorscale='RdBu',
                                        reversescale=True,
                                        color=[],
                                        size=15,
                                        colorbar=dict(thickness=10,
                                                      title='Node Connections',
                                                      xanchor='left',
                                                      titleside='right'),
                            line=dict(width=0)))
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
        
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="No. of connections",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig


activation_functions = [sigmoid, tanh, sine, square, cosine, identity]
