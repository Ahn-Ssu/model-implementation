# 1. Model 
args.model = GCN
args.input_dim = utils.chem.n_atom_feats
args.hidden_dim = 128
args.out_dim = 1
args.head_num = 4
args.ex_dim = 20
args.convNorm = False
args.skipConcat = True
args.skipCompress = False
args.convAggr = 'add'

# 2. learing 
args.batch_size = 17
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 100000

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

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

