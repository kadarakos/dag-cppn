import random
import math
import secrets

class Node(object):
    def __init__(self, idx, func):
        self.idx = idx
        self.func = func
        self.inputs = []
        self.out_edges = []

    def add_outedge(self, edge):
        self.out_edges.append(edge)

    def add_input(self, x):
        self.inputs.append(x)

    def propagate(self):
        x = self.func(sum(self.inputs))
        self.inputs = []
        return x

    def __str__(self):
        return "{}-{}".format(self.idx, str(self.func.__name__))


class Edge(object):
    def __init__(self, idx, parent, child, weight):
        self.idx = idx
        self.parent = parent
        self.child = child
        self.weight = weight

    def propagate(self):
        x = self.parent.propagate()
        x += self.weight
        self.child.add_input(x)


class Network(object):
    def __init__(self):
        self.layers = [[], [], []]
        self.nodes = []
        self.edges = []
        self.networkx_graph = None

    def gen_weight(self, mu=0, sigma=1):
        w = random.gauss(mu, sigma)
        return w

    def gen_node(self, layer):
        node_idx = "{}-{}".format(layer, len(self.nodes))
        f = random.choice(activation_functions)
        return Node(node_idx, f)

    def add_node(self, node, layer):
        self.nodes.append(node)
        self.layers[layer].append(node)

    def add_edge(self, parent, child):
        w = self.gen_weight()
        edge = Edge("bubu", parent, child, w)
        parent.add_outedge(edge)
        self.edges.append(edge)

    def add_layer(self):
        layers = [[]] * (len(self.layers) + 1)
        for i, l in enumerate(self.layers[:-1]):
            layers[i] = l
        layers[-1] = self.layers[-1]
        self.layers = layers
    
    def add_random_node(self):
        l = random.randint(1, len(self.layers) - 1)
        n = gen_node(l)
        self.add_node(n, l)


    def add_random_edge(self, weight=False):
        if not weight:
            weight = gen_weight()
        # Child layer
        layer2 = random.randint(0, len(self.layers) - 2)
        # Parent layer
        layer1 = random.randint(0, layer2)
        edge_idx = "{}-{}-{}".format(layer1,
                                    layer2,
                                    len(self.edges))
        parent = random.choice(self.layers[layer1])
        child = random.choice(self.layers[layer2])
        edge = Edge(edge_idx, parent, child, weight)
        parent.add_outedge(edge)
        self.edges.append(edge)

    # Add Input nodes, hidden nodes and an output node
    # Each hidden node is connected to the output
    # Each input node can be either connected to the hidden or output or both.
    # Make sure each hidden node has at least one input node connected
    def initialize(self, n_hidden=False):
        if not n_hidden:
            n_hidden = random.randint(1, 10)
        # Add output node, calues between 0-1
        output_node = Node('output', sigmoid)
        self.add_node(output_node, 2)
        # Add input nodes
        input_nodes = []
        for i, idx in enumerate(['x', 'y', 'r']):
            n = Node(idx, random.choice(activation_functions))
            self.add_node(n, 0)
        # Add hidden nodes and connect to output
        for i in range(n_hidden):
            n = self.gen_node(1)
            self.add_node(n, 1)
            self.add_edge(n, output_node)
        #Connect each input node to some higher node
        #To make sure the output node is reachable
        for n in self.layers[0]:
            target_node = random.choice(self.layers[1] + self.layers[2])
            self.add_edge(n, target_node)
        #If there is a remaining hidden nodes not connected to
        #input do connect it
        connected_hidden = set()
        for n in self.layers[0]:
            for e in n.out_edges:
                connected_hidden.add(e.child)
        empty_hidden = set(self.layers[1]) - connected_hidden
        for n in empty_hidden:
            source = random.choice(self.layers[0])
            self.add_edge(source, n)

    def forward_propagate(self, x, y, r):
        self.layers[0][0].add_input(x)
        self.layers[0][1].add_input(y)
        self.layers[0][2].add_input(r)
        for l in self.layers:
            for n in l:
                for e in n.out_edges:
                    e.propagate()
        return self.layers[-1][0].propagate()

    # Might never even do this :D
    def backward_propagate():
        raise NotImplementedError

if __name__=="__main__":
    network = Network()
    network.initialize(n_hidden=12)
    for i, l in enumerate(network.layers):
        for j, n in enumerate(l):
            print(i, j, n)
    
    for i, l in enumerate(network.layers):
        for j, n in enumerate(l):
            for e in n.out_edges:
                print(e.parent, e.weight, e.child)

    network.forward_propagate(0.1,0.2,0.3)

    img_size = 256
    image = np.zeros((img_size, img_size))
    scale = 0.2
    factor = img_size/scale
    for i in range(img_size):
        for j in range(img_size):
            x = i/factor - 0.5 * scale
            y = j/factor - 0.5 * scale
            r = math.sqrt(x**2 + y**2)
            image[i,j] = network.forward_propagate(x, y, r)
    print(image)
