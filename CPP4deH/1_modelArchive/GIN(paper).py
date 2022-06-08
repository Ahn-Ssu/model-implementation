import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geonn
from skipNetwork import GatedSkipConnection, SkipConnection


# 5 GNN layers (including the input layer) are applied, and all MLPs have 2 layers.
# Batch normalization (Ioffe & Szegedy, 2015) is applied on every hidden layer.

# nn.Sequential(
        #       nn.Linear(args.hidden_dim, args.hidden_dim + int(args.hidden_dim * 0.1)),
        #       nn.ReLU(),
        #       nn.Linear(args.hidden_dim + int(args.hidden_dim * 0.1), args.hidden_dim + int(args.hidden_dim * 0.1)),
        #       nn.ReLU(),
        #       nn.Linear(args.hidden_dim + int(args.hidden_dim * 0.1), args.hidden_dim),
        #       nn.ReLU(),
        #   )

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()

        self.conv1 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.input_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv3 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv4 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv5 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
            )
        )

        self.fc1 = nn.Linear(args.hidden_dim*5 + args.input_dim, args.hidden_dim*5 + args.input_dim)
        self.fc3 = nn.Linear(args.hidden_dim*5 + args.input_dim, args.out_dim)

    def forward(self, g):

        h1 = self.conv1(g.x, g.edge_index)
        h2 = self.conv2(h1, g.edge_index)
        h3 = self.conv3(h2, g.edge_index)
        h4 = self.conv4(h3, g.edge_index)
        h5 = self.conv5(h4, g.edge_index)

        hg = torch.cat([g.x, h1, h2, h3, h4, h5],
                       dim=1)
        # Read Out Layer 
        hg = geonn.global_add_pool(hg, g.batch)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
