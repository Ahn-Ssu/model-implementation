"""
Na, G. S., Kim, H. W., & Chang, H. (2020). Costless performance improvement in machine learning for graph-based molecular analysis.
Journal of Chemical Information and Modeling, 60(3), 1137-1145. https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00816 
"""
class eGCN(nn.Module):
    def __init__(self, args):
        super(eGCN, self).__init__()

        self.hidden_dim = args.hidden_dim
        
        self.conv1 = geonn.GCNConv(args.input_dim, self.hidden_dim)
        self.conv2 = geonn.GCNConv(self.hidden_dim, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim + args.ex_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, args.out_dim)

    def forward(self, g):
        h1 = self.conv1(g.x, g.edge_index)
        h1 = F.relu(h1)

        h2 = self.conv2(h1, g.edge_index)
        h2 = F.relu(h2)

        hg = geonn.global_mean_pool(h2, g.batch)

        # extend Strcture(molecule) Feature
        hg = torch.cat([hg, g.eFeature],
                       dim=1)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out

