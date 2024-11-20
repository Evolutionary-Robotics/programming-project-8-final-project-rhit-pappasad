import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import networkx as nx

@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@jit(nopython=True)
def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def identity(x):
    return x


def step(x):
    return np.heaviside(x)

@jit(nopython=True)
def fastDot(a, w, b):
    return np.dot(a, w) + b

class NeuralNetwork:
    _act = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh,
        'identity': identity,
        'step': step
    }
    _act_deriv = {
        'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
        'relu': lambda x: np.heaviside(x, 1),
        'tanh': lambda x: 1 - np.tanh(x)**2,
        'identity': lambda x: 1,
        'step': lambda x: np.inf if x == 0 else 0
    }

    def __init__(self, num_inputs, hidden_map, num_outputs, activation, oa, weight_range=1, bias_range=1):
        self.num_hidden = len(hidden_map)
        self.num_layers = self.num_hidden + 2
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_map = hidden_map

        self.units_per_layer = np.zeros(self.num_layers).astype(int)
        self.units_per_layer[0] = num_inputs
        self.units_per_layer[1:-1] = hidden_map
        self.units_per_layer[-1] = num_outputs

        self.hidden_act = self._act[activation]
        self.output_act = self._act[oa]

        self.weights = []
        self.biases = []
        self.weight_range = weight_range
        self.bias_range = bias_range

    def setParams(self, params):
        self.weights = []
        self.biases = []
        start = 0
        for layer in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[layer]*self.units_per_layer[layer+1]
            self.weights.append((params[start:end]*self.weight_range).reshape(self.units_per_layer[layer], self.units_per_layer[layer+1]).astype(np.float32))
            start = end
        start = 0
        for layer in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[layer+1]
            self.biases.append((params[start:end]*self.bias_range).reshape(1, self.units_per_layer[layer+1]).astype(np.float32))
            start = end

    def forward(self, inputs):
        a = np.array(inputs).astype(np.float32)
        for layer in np.arange(self.num_layers - 2):
            z = fastDot(a, self.weights[layer], self.biases[layer])
            a = self.hidden_act(z)
        z = fastDot(a, self.weights[-1], self.biases[-1])
        a = self.output_act(z)
        return a.flatten()

    def visualize(self, title, save=False):
        G = nx.DiGraph()

        inputs = [f'X[{i-1}]' for i in range(self.num_inputs, 0, -1)]
        G.add_nodes_from(inputs)

        hidden = []
        for l in range(self.num_hidden):
            layer_nodes = [f'H{l+1}[{i-1}]' for i in range(self.hidden_map[l], 0, -1)]
            hidden.append(layer_nodes)
            G.add_nodes_from(layer_nodes)

        outputs = [f'Y[{i-1}]' for i in range(self.num_outputs, 0, -1)]
        G.add_nodes_from(outputs)

        for input_node in inputs:
            for hidden_node in hidden[0]:
                G.add_edge(input_node, hidden_node)

        for i in range(self.num_hidden - 1):
            for node in hidden[i]:
                for next_node in hidden[i + 1]:
                    G.add_edge(node, next_node)

        for hidden_node in hidden[-1]:
            for output_node in outputs:
                G.add_edge(hidden_node, output_node)

        # Define positions for nodes (centered vertical layout)
        pos = {}
        x_offset = 0
        y_offset = 0

        # Input layer (centered)
        for i, node in enumerate(inputs):
            pos[node] = (x_offset, i - self.num_inputs // 2)

        # Hidden layers (centered)
        for layer_index, layer in enumerate(hidden):
            x_offset += 2
            layer_size = len(layer)
            #y_offset += 0.5 * (layer_size - self.units_per_layer[layer_index])

            for i, node in enumerate(layer):
                pos[node] = (x_offset, y_offset + i - layer_size // 2)

        # Output layer (centered)
        x_offset += 2
        for i, node in enumerate(outputs):
            pos[node] = (x_offset, i - self.num_outputs // 2)

        #plt.figure(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, font_weight="bold",
                arrows=False)
        #plt.title(title)
        if save:
            plt.savefig(f'{title}Network.png', dpi=300)
        plt.show()












