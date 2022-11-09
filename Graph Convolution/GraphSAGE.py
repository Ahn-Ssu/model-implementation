"""
Hamilton, W., Ying, Z., & Leskovec, J. (2017). 
Inductive representation learning on large graphs. Advances in neural information processing systems, 30.
https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html 
"""

class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = geonn.SAGEConv(args.input_dim, args.hidden_dim)
        self.norm1 = geonn.BatchNorm(args.hidden_dim)

        self.conv2 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm2 = geonn.BatchNorm(args.hidden_dim)

        self.conv3 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm3 = geonn.BatchNorm(args.hidden_dim)

        self.conv4 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm4 = geonn.BatchNorm(args.hidden_dim)
        
        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.out_dim)

    def forward(self, g):
        # Conv Layer 1
        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)

        # Conv Layer 2
        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)

        # Conv Layer 3
        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)

        # Conv Layer 4
        h4 = self.conv4(h3, g.edge_index)
        h4 = self.norm4(h4)
        h4 = F.relu(h4)

        # Read Out Layer 
        hg = geonn.global_mean_pool(h4, g.batch)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
