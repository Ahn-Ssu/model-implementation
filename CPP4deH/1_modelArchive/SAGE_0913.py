# 9월 13일 제공 모델 

# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")

# 1. Model
# 1. Model 
args.model = SAGE
args.input_dim = utils.chem.n_atom_feats
args.hidden_dim = 128
args.out_dim = 1
args.head_num = 4
args.ex_dim = 20
args.convNorm = False
args.skipConcat = True
args.skipCompress = False

# 2. learing 
args.batch_size = 17
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 50000


class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()

        self.hidden_dim = args.hidden_dim
        
        self.conv1 = geonn.SAGEConv(args.input_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm1 = geonn.BatchNorm(self.hidden_dim)
        self.skip1 = SkipConnection(args.input_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv2 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm2 = geonn.BatchNorm(self.hidden_dim)
        self.skip2 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv3 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm3 = geonn.BatchNorm(self.hidden_dim)
        self.skip3 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv4 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm4 = geonn.BatchNorm(self.hidden_dim)
        self.skip4 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv5 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm5 = geonn.BatchNorm(self.hidden_dim)
        self.skip5 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv6 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm)
        self.norm6 = geonn.BatchNorm(self.hidden_dim)
        self.skip6 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)
        
        self.hidden_dim *= 2
        self.fc1 = nn.Linear(self.hidden_dim + args.ex_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, args.out_dim)

    def forward(self, g):
        # Conv Layer 1
        id1 = g.x
        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.skip1(id1, h1)

        # Conv Layer 2
        id2 = h1
        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.skip2(id2, h2)


        # Conv Layer 3
        id3 = h2
        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = self.skip3(id3, h3)


        # Conv Layer 4
        id4 = h3
        h4 = self.conv4(h3, g.edge_index)
        h4 = self.norm4(h4)
        h4 = F.relu(h4)
        h4 = self.skip4(id4, h4)

        # Conv Layer 5
        id5 = h4
        h5 = self.conv5(h4, g.edge_index)
        h5 = self.norm5(h5)
        h5 = F.relu(h5)
        h5 = self.skip5(id5, h5)

        # Conv Layer 6
        id6 = h5
        h6 = self.conv6(h5, g.edge_index)
        h6 = self.norm6(h6)
        h6 = F.relu(h6)
        h6 = self.skip6(id6, h6)

        # Read Out Layer 
        hg = geonn.global_mean_pool(h6, g.batch)

        # extend Strcture(molecule) Feature
        hg = torch.cat([hg, g.eFeature],
                       dim=1)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out

