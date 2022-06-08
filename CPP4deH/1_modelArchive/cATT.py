import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geonn
from skipNetwork import GatedSkipConnection, SkipConnection

class ATT(nn.Module):
    def __init__(self, args):
        super(ATT, self).__init__()
        
        #for summation code
        # self.conv1 = geonn.GATConv(args.input_dim, args.hidden_dim, heads=args.head_num, concat=False)
        # self.conv2 = geonn.GATConv(args.hidden_dim, args.hidden_dim, heads=args.head_num, concat=False)
        # self.conv3 = geonn.GATConv(args.hidden_dim, args.hidden_dim, heads=args.head_num, concat=False)

        #for mormal concat
        # self.conv1 = geonn.GATConv(args.input_dim, args.hidden_dim, heads=args.head_num)
        # self.conv2 = geonn.GATConv(int(args.hidden_dim*args.head_num), args.hidden_dim, heads=args.head_num)
        # self.conv3 = geonn.GATConv(int(args.hidden_dim*args.head_num), args.hidden_dim, heads=args.head_num)
        # args.hidden_dim = int(args.hidden_dim*args.head_num)

        # for same input/output concat
        self.conv1 = geonn.GATConv(args.input_dim, int(args.hidden_dim/args.head_num), heads=args.head_num)
        self.conv2 = geonn.GATConv(args.hidden_dim, int(args.hidden_dim/args.head_num), heads=args.head_num)
        self.conv3 = geonn.GATConv(args.hidden_dim, int(args.hidden_dim/args.head_num), heads=args.head_num)

        self.skip1 = GatedSkipConnection(args.input_dim, args.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)
        self.skip2 = GatedSkipConnection(args.hidden_dim, args.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)
        self.skip3 = GatedSkipConnection(args.hidden_dim, args.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.norm1 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)
        self.norm2 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)
        self.norm3 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)

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

        # Read Out Layer 
        hg = geonn.global_mean_pool(h3, g.batch)

        # extend Strcture(molecule) Feature
        hg = torch.cat([hg, g.eFeature],
                       dim=1)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out