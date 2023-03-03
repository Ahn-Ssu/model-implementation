import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        # WRITE YOUR CODE HERE
        h = self.linear_input(x) + self.linear_hidden(state[0])

        chunk_forgetgate, chunk_ingate, chunk_cellgate, chunk_outgate = torch.chunk(h, chunks=4, dim=1)

        f_x = torch.sigmoid(chunk_forgetgate)
        i_x = torch.sigmoid(chunk_ingate)
        c_y = torch.tanh(chunk_cellgate)
        o_x = torch.sigmoid(chunk_outgate)

        cx = f_x * state[1]
        cy = cx + i_x * c_y
        hy = o_x * torch.tanh(cy)

        return hy, (hy, cy)