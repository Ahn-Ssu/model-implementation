
# 7월 31일자 학습 제공 모델 
class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()
        
        self.conv1 = geonn.SAGEConv(args.input_dim, args.hidden_dim)
        self.norm1 = geonn.BatchNorm(args.hidden_dim)
        self.skip1 = SkipConnection(args.input_dim, args.hidden_dim, concat=True, compress=False)

        args.hidden_dim *= 2
        self.conv2 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm2 = geonn.BatchNorm(args.hidden_dim)
        self.skip2 = SkipConnection(args.hidden_dim, args.hidden_dim, concat=True, compress=False)

        args.hidden_dim *= 2
        self.conv3 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm3 = geonn.BatchNorm(args.hidden_dim)
        self.skip3 = SkipConnection(args.hidden_dim, args.hidden_dim, concat=True, compress=False)

        args.hidden_dim *= 2
        self.conv4 = geonn.SAGEConv(args.hidden_dim, args.hidden_dim)
        self.norm4 = geonn.BatchNorm(args.hidden_dim)
        self.skip4 = SkipConnection(args.hidden_dim, args.hidden_dim, concat=True, compress=False)
        
        args.hidden_dim *= 2
        self.fc1 = nn.Linear(args.hidden_dim + args.ex_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.out_dim)

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

        # Read Out Layer 
        hg = geonn.global_mean_pool(h4, g.batch)

        # extend Strcture(molecule) Feature
        hg = torch.cat([hg, g.eFeature],
                       dim=1)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
