import torch

def message_passing(nodes, edges, message_fn, edge_features = None):

    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \sum_{i \in N(j)} f(h_i^{t-1}, h_j^{t-1})
    :param nodes: (n_nodes, n_features) tensor of node hidden states.
    :param edges: (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    :param edge_features: (n_edges, n_edge_features) features for each edge. Set to zero if the edges don't have features.
    :param message_fn: message function, will be called with input of shape (n_edges, 2*n_features + edge_features).
                       The output shape is (n_edges, n_outputs), where you decide the size of n_outputs
    :return: (n_nodes, n_output) Sum of messages arriving at each node.
    """

    n_nodes = nodes.shape[0]
    n_features = nodes.shape[1]
    n_edges = edges.shape[0]

    # For each edge, features for the start and end node
    from_edges = edges[:,0]
    to_edges = edges[:,1]

    # (n_edges, n_features)
    from_node_features = nodes[from_edges]
    to_node_features = nodes[to_edges]

    # (n_edges, 2 * n_node_features)
    features = torch.cat([from_node_features, to_node_features], dim = -1)

    # Concatenate edge features if they exist
    if not edge_features is None:
        # (n_edges, 2 * n_node_features + n_edge_features)
        features = torch.cat([features, edge_features], dim = -1)

    # (n_edges, n_outputs)
    messages = message_fn(features)

    n_output = messages.shape[1]
    out_shape = (n_nodes, n_output)

    # Updates for nodes at to_edges corresponding to messages
    # Only need to_edges, to avoid double counting (assuming symmetry in from <-> to)
    updates = torch.zeros(out_shape)

    source_idxs = torch.arange(0, n_edges)
    destination_idxs = to_edges

    # Sum over input edges -> sum those with repeated destination_idxs
    # PyTorch's scatter_add does NOT have the same behaviour as TensorFlow: In PyTorch ALL elements
    # of the source tensor are summed, rather than just those with repeated indices

    for source_idx in source_idxs:
        updates[destination_idxs[source_idx]] += messages[source_idx]

    return updates

if __name__ == '__main__':

    import numpy as np

    x = np.random.randn(3, 2).astype(np.float32)
    nodes = torch.tensor(x, dtype = torch.float32)
    edges = torch.tensor(np.array([[0, 1], [1, 2], [2, 1]]), dtype = torch.int64)
    edge_features = torch.zeros((3, 1), dtype = torch.float32)

    def message_fn(x):
        return x[:, 0:2] + x[:, 2:4]

    out = message_passing(nodes, edges, message_fn, edge_features)
    expected = np.array([
        [0, 0],  # no messages for node 0
        (x[0] + x[1]) + (x[2] + x[1]),  # 0 to 1, and 2 to 1
        (x[1] + x[2])  # 1 to 2
    ], dtype=np.float32)

    print('Pass:', np.all(np.isclose(np.array(out), np.array(expected))))
