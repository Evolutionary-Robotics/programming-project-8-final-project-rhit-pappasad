import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle
import os

ACTIVATION_FUNCTIONS = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'softmax': F.softmax,
    'identity': lambda x: x,
    'step': lambda x: (x > 0).float(),
    'leaky_relu': lambda x: F.leaky_relu(x, negative_slope=0.01)
}

class NeuralNetwork(nn.Module):
    __device__ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, num_inputs, hidden_map, num_outputs, hidden_activation, output_activations, w_range=1, b_range=1):
        super(NeuralNetwork, self).__init__()

        self.num_layers = len(hidden_map) + 2
        self.num_hiddens = len(hidden_map)
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs

        try:
            self._ha = hidden_activation.lower()
            self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation.lower()]
            self._oa = output_activations
            if isinstance(output_activations, str):
                self.output_activations = [ACTIVATION_FUNCTIONS[output_activations.lower()]]*num_outputs
            else:
                self.output_activations = [ACTIVATION_FUNCTIONS[act.lower()] for act in output_activations]
        except KeyError:
            print(f"<<<ERROR>>>   network -> network.py -> NeuralNetwork.init(): {hidden_activation} or {output_activations} not valid")
            sys.exit()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.num_inputs, hidden_map[0]))
        for i in range(1, len(hidden_map)):
            self.layers.append(nn.Linear(hidden_map[i-1], hidden_map[i]))
        self.layers.append(nn.Linear(hidden_map[-1], self.num_outputs))

        self.init_weights(w_range, b_range)

    def init_weights(self, w_range, b_range):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -w_range, w_range)
            nn.init.uniform_(layer.bias, -b_range, b_range)

    def forward(self, inputs):
        a = torch.tensor(inputs, dtype=torch.float16).to(self.__device__)
        for i in range(len(self.layers) - 1):
            a = self.hidden_activation(self.layers[i](a))
        a = self.layers[-1](a)

        outputs = [self.output_activations[i](a[i]).tolist() for i in range(self.num_outputs)]
        #outputs = torch.stack([self.output_activations[i](a[:, i]) for i in range(self.num_outputs)], dim=1)
        return outputs

    def setParams(self, params):
        params = torch.tensor(params, dtype=torch.float16).to(self.__device__)
        start = 0
        for layer in self.layers:
            w_shape = layer.weight.shape
            b_shape = layer.bias.shape

            w_end = start + w_shape.numel()
            b_end = w_end + b_shape.numel()

            layer.weight.data = params[start:w_end].reshape(w_shape).to(layer.weight.device)
            layer.bias.data = params[w_end:b_end].reshape(b_shape).to(layer.bias.device)

            start = b_end

    def save(self, name):
        info = {
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'hidden_map': [layer.out_features for layer in self.layers[:-1]],
            'hidden_activation': self._ha,
            'output_activations': self._oa,
            'state_dict': self.state_dict()
        }
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved', name+'.pkl')
        with open(path, 'wb') as f:
            pickle.dump(info, f)
        print(f"Network {name} saved at {path}")

    @classmethod
    def load(cls, name):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved', name+'.pkl')
        with open(path, 'rb') as f:
            info = pickle.load(f)

        state = info.pop('state_dict')
        network = cls(**info)
        network.load_state_dict(state)
        network.to(cls.__device__)
        print(f"Loaded {name} from {path}")
        return network








if __name__ == '__main__':
    h_a = 'leaky_relu'
    h_o = ['sigmoid', 'tanh']
    hidden_map = [5, 3, 8, 4]
    inputs = [1, 2, 3, 4]

    net = NeuralNetwork.load('test')#NeuralNetwork(len(inputs), hidden_map, len(h_o), h_a, h_o)
    #print(sum([p.numel() for p in net.parameters()]))
    params = [0.1]*sum(p.numel() for p in net.parameters())
    net.setParams(params)
    output = net(inputs)
    #net.save('test')
    print(output)


