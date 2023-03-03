"""
Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks?. 
arXiv preprint arXiv:1810.00826. https://arxiv.org/abs/1810.00826 
"""

# 5 GNN layers (including the input layer) are applied, and all MLPs have 2 layers.
# Batch normalization (Ioffe & Szegedy, 2015) is applied on every hidden layer.

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.hidden_dim = args.hidden_dim

        self.conv1 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.input_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
        )
        self.conv2 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
        )
        self.conv3 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
        )
        self.conv4 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
        )
        self.conv5 = geonn.GINConv(
            nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
        )


        self.norm1 = geonn.BatchNorm(self.hidden_dim)
        self.norm2 = geonn.BatchNorm(self.hidden_dim)
        self.norm3 = geonn.BatchNorm(self.hidden_dim)
        self.norm4 = geonn.BatchNorm(self.hidden_dim)
        self.norm5 = geonn.BatchNorm(self.hidden_dim)

        self.fc1 = nn.Linear(args.hidden_dim*5 + args.input_dim, args.hidden_dim*5 + args.input_dim)
        self.fc3 = nn.Linear(args.hidden_dim*5 + args.input_dim, args.out_dim)

    def forward(self, g):

        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)

        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)

        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)

        h4 = self.conv4(h3, g.edge_index)
        h4 = self.norm4(h4)
        h4 = F.relu(h4)

        h5 = self.conv5(h4, g.edge_index)
        h5 = self.norm5(h5)
        h5 = F.relu(h5)

        hg = torch.cat([g.x, h1, h2, h3, h4, h5],
                       dim=1)

        # Read Out Layer 
        hg = geonn.global_mean_pool(hg, g.batch)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out