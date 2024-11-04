def draw_centered_feedforward_network(inputs, outputs, hidden_layers, title):
    G = nx.DiGraph()

    # Add input layer nodes
    input_nodes = [f'I{i+1}' for i in range(inputs)]
    G.add_nodes_from(input_nodes)

    # Add hidden layer nodes
    hidden_nodes = []
    for layer_index, layer_size in enumerate(hidden_layers):
        layer_nodes = [f'H{layer_index+1}_{i+1}' for i in range(layer_size)]
        hidden_nodes.append(layer_nodes)
        G.add_nodes_from(layer_nodes)

    # Add output layer nodes
    output_nodes = [f'O{i+1}' for i in range(outputs)]
    G.add_nodes_from(output_nodes)

    # Add edges from input to first hidden layer
    for input_node in input_nodes:
        for hidden_node in hidden_nodes[0]:
            G.add_edge(input_node, hidden_node)

    # Add edges between hidden layers
    for i in range(len(hidden_nodes) - 1):
        for node in hidden_nodes[i]:
            for next_node in hidden_nodes[i + 1]:
                G.add_edge(node, next_node)

    # Add edges from last hidden layer to output layer
    for hidden_node in hidden_nodes[-1]:
        for output_node in output_nodes:
            G.add_edge(hidden_node, output_node)

    # Define positions for nodes (centered vertical layout)
    pos = {}
    x_offset = 0

    # Input layer (centered)
    for i, node in enumerate(input_nodes):
        pos[node] = (x_offset, i - inputs // 2)

    # Hidden layers (centered)
    for layer_index, layer in enumerate(hidden_nodes):
        x_offset += 2
        layer_size = len(layer)
        for i, node in enumerate(layer):
            pos[node] = (x_offset, i - layer_size // 2)

    # Output layer (centered)
    x_offset += 2
    for i, node in enumerate(output_nodes):
        pos[node] = (x_offset, i - outputs // 2)

    # Draw the network
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, font_weight="bold", arrows=False)
    plt.title(title)
    plt.show()

# Draw the network with centered layers
draw_centered_feedforward_network(6, 2, [4, 8, 4, 2], "Centered Network with 4 hidden layers (4, 8, 4, 2 neurons)")
