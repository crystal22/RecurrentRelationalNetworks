import torch

from message_passing import message_passing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# From RRN paper repository:
# https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/master/tasks/sudoku/rrn.py

def sudoku_edges():
    def cross(a):
        return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

    idx = np.arange(81).reshape(9, 9)
    rows, columns, squares = [], [], []
    for i in range(9):
        rows += cross(idx[i, :])
        columns += cross(idx[:, i])
    for i in range(3):
        for j in range(3):
            squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
    return list(set(rows + columns + squares))

class RRN(torch.nn.Module):
    """Recurrent relational-network"""

    def __init__(self, linear_size = 32, lstm_size = 32, embed_size = 32, message_size = 32, batch_size = 1024):

        super(RRN, self).__init__()

        self.batch_size = batch_size
        self.embed_size = embed_size
        self.linear_size = linear_size
        self.lstm_size = lstm_size
        self.message_size = message_size

        self.edges = sudoku_edges
        seld.n_nodes = 81 # batch? 0 axis?

        self.edge_indices = torch.tensor(
            [(i + (b * 81), j + (b * 81))
                for b in range(self.batch_size)
                    for i, j in self.edges],
            torch.int32)

        self.n_edges = self.edge_indices.shape[0]

        self.edge_features = None
        self.n_edge_features = 0

        # (batch_size, 9 * 9, 2) # Why 2?
        self.positions = torch.tensor(
            [[(i, j) for i in range(9) for j in range(9)]
                for b in range(self.batch_size)], torch.int32)

        # (batch_size, 9 * 9, embed_size)
        self.rows = torch.nn.Embedding(9, self.embed_size)

        # (batch_size, 9 * 9, embed_size)
        self.cols = torch.nn.Embedding(9, self.embed_size)

        # 10 -> 9 values + empty
        self.initial_state = torch.nn.Embedding(10, self.embed_size)

        # Maybe a separate module for these

        self.pre_layers = [torch.nn.Linear(3 * self.embed_size, self.linear_size)]
        self.pre_layers.extend([torch.nn.Linear(self.linear_size, self.linear_size) for i in range(3)])

        # Input of (n_edges, 2 * n_node_features + n_edge_features)
        # Input of (n_edges, 2 * linear_size + n_edge_features)
        # n_edges, n_nodes -> ???? 0 dimension is confusing...
        self.message_passing_layers = [torch.nn.Linear(2 * self.linear_size + self.n_edge_features, self.message_size)]
        self.message_passing_layers.extend([torch.nn.Linear(self.message_size, self.message_size) for i in range(3)])

        # Input of shape (n_nodes, message_size + linear_size)
        # What about n_nodes + batch dimension? can't pass 2d...
        # what is done in TF?
        self.post_layers = [torch.nn.Linear(self.message_size + self.linear_size, self.linear_size)]
        self.post_layers.extend([torch.nn.Linear(self.linear_size, self.linear_size) for i in range(3)])

        # After post_layers, so input size is linear_size
        self.node_lstm = torch.nn.LSTMCell(self.linear_size, self.lstm_size)

        # 9 outputs in paper, 10 in reference implementation?
        self.output_layer = torch.nn.Linear(self.lstm_size, 9)

    def message_function(self, x):

        for i in range(3):
            x = torch.nn.functional.relu(self.message_passing_layers[i](x))
        x = self.message_passing_layers[3](x)

        return x

    def forward(self, x):

        x = self.initial_state(x)
        r = self.rows(positions[:, :, 0])
        c = self.cols(positions[:, :, 1])

        x = torch.cat([x, r, c], dim = -1)

        x = x.reshape((-1, 3 * self.embed_size))

        for i in range(3):
            x = torch.nn.functional.relu(self.pre_layers[i](x))
        x = self.pre_layers[3](x)

        x0 = x.clone()
        n_nodes = x.shape[0] # batch dim?

        # Zeros of shape (batch_size, hidden_size)
        h_state = torch.zeros((x.shape[0], self.lstm_size))
        c_state = torch.zeros((x.shape[0], self.lstm_size))

        for step in range(self.n_steps):
            # Pass member function (okay?)
            x = message_passing(x, self.edge_indices, self.edge_features, self.message_function)

            x = torch.cat([x, x0], dim = -1)

            for i in range(3):
                x = torch.nn.functional.relu(self.post_layers[i](x))
            x = self.post_layers[3](x)

            # Call lstm (for memory)
            h_state, c_state = self.node_lstm(x, (h_state, c_state))

            # So that we can operate on this, but pass the actual state to the next step of the LSTM
            x = c_state.clone()

            # Reference implementation averages over losses at each of these steps, rather
            # that calculating gradients from the final loss.
            # The paper says this should be MINIMISED at every step, though.

        # Input to torch.nn.CrossEntropyLoss, roughly equivalent to TF softmax_cross_entropy_with_logits
        x = self.output_layer(x).reshape((-1, 81, 9))

        return x

    def eval(self, x):
        return torch.nn.functional.softmax(forward(x), dim = -1)

def train(net, x, y, learning_rate = 0.01, epochs = 100):

    loss_function = torch.nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        train_outputs = net(x)

        # Calculate gradients and loss from final outputs, rather than over LSTM steps

        # Outputs with reduction = 'none' are same shape as the input, (N, n_outputs)
        loss = loss_function(train_outputs, y)

        # We want to sum over outputs (per paper)...
        loss = torch.sum(loss, dim = -1)

        # ...and average over samples (per usual)
        loss = torch.mean(loss, dim = 0)

        # Loss is now a scalar
        loss.backward()

        optimizer.step()

        losses.append(loss)

    return losses
