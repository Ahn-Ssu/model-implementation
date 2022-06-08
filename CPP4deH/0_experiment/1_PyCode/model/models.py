import sys
sys.path.append("/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/model")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geonn
from skipNetwork import GatedSkipConnection, SkipConnection
from modify_sage_conv import mSAGEConv
from graphConv_1x1 import graphConv_1x1

#  extend하는 채널수도 받게 하장 
#  구조는 동일하고 conv layer가 바뀌니 이걸 좀 자동화 해보장 class GCN(nn.Module):
class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.hidden_dim = args.hidden_dim
        
        self.conv1 = geonn.GCNConv(args.input_dim, self.hidden_dim)
        self.conv2 = geonn.GCNConv(self.hidden_dim, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, args.out_dim)

    def forward(self, g):
        h1 = self.conv1(g.x, g.edge_index)
        h1 = F.relu(h1)

        h2 = self.conv2(h1, g.edge_index)
        h2 = F.relu(h2)


        # original structure는 두번째 conv2 이후에 reLU가 아닌 softmax에 집어넣고 아웃풋을 만들어냄 
        # 아래 구조는 동일한 interpreter를 갖게 하고, regression을 위해서 생성
        hg = geonn.global_mean_pool(h2, g.batch) 

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
        
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

        # self.conv1 = geonn.SAGEConv(args.input_dim, self.hidden_dim)#, aggr=args.convAggr)
        # self.conv2 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)#, aggr=args.convAggr)
        # self.conv3 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)#, aggr=args.convAggr)
        # self.conv4 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)#, aggr=args.convAggr)
        # self.conv5 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)#, aggr=args.convAggr)
        # self.conv6 = mSAGEConv(self.hidden_dim, self.hidden_dim, aggr=args.convAggr)
        # self.conv5 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)#, aggr=args.convAggr)

        self.norm1 = geonn.BatchNorm(self.hidden_dim)
        self.norm2 = geonn.BatchNorm(self.hidden_dim)
        self.norm3 = geonn.BatchNorm(self.hidden_dim)
        self.norm4 = geonn.BatchNorm(self.hidden_dim)
        self.norm5 = geonn.BatchNorm(self.hidden_dim)
        # self.norm6 = geonn.BatchNorm(self.hidden_dim)

        # self.skip1 = GatedSkipConnection(args.input_dim, args.hidden_dim)
        # self.skip2 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)
        # self.skip3 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)
        # self.skip4 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)
        # self.skip5 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)
        # self.skip6 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)




        self.fc1 = nn.Linear(args.hidden_dim*5 +args.input_dim, args.hidden_dim*5 +args.input_dim)
        self.fc3 = nn.Linear(args.hidden_dim*5 +args.input_dim, args.out_dim)

    def forward(self, g):

        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        # h1 = self.skip1(g.x, h1)

        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        # h2 = self.skip2(h1, h2)

        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        # h3 = self.skip3(h2, h3)


        h4 = self.conv4(h3, g.edge_index)
        h4 = self.norm4(h4)
        h4 = F.relu(h4)
        # h4 = self.skip4(h3, h4)


        h5 = self.conv5(h4, g.edge_index)
        h5 = self.norm5(h5)
        h5 = F.relu(h5)
        # h5 = self.skip5(h4, h5)

        # h6 = self.conv6(h5, g.edge_index)
        # h6 = self.norm6(h6)
        # h6 = F.relu(h6)
        # h6 = self.skip6(h5, h6)


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





class SAGIN(nn.Module):
    def __init__(self, args):
        super(SAGIN, self).__init__()

        self.hidden_dim = args.hidden_dim
        
        self.conv1 = mSAGEConv(args.input_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm1 = geonn.BatchNorm(self.hidden_dim)
        self.skip1 = SkipConnection(args.input_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv2 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm2 = geonn.BatchNorm(self.hidden_dim)
        self.skip2 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv3 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm3 = geonn.BatchNorm(self.hidden_dim)
        self.skip3 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)


        self.hidden_dim *= 2
        self.redconv1 = nn.Linear(self.hidden_dim, int(2*args.hidden_dim), bias=False)
        self.hidden_dim = int(2*args.hidden_dim)
        # self.bottlenorm = geonn.BatchNorm(self.hidden_dim)

        self.conv4 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm4 = geonn.BatchNorm(self.hidden_dim)
        self.skip4 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv5 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm5 = geonn.BatchNorm(self.hidden_dim)
        self.skip5 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv6 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm6 = geonn.BatchNorm(self.hidden_dim)
        self.skip6 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.conv7 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        self.norm7 = geonn.BatchNorm(self.hidden_dim)
        self.skip7 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        # self.hidden_dim *= 2
        # self.conv8 = mSAGEConv(self.hidden_dim, self.hidden_dim, normalize=args.convNorm, aggr=args.convAggr)
        # self.norm8 = geonn.BatchNorm(self.hidden_dim)
        # self.skip8 = SkipConnection(self.hidden_dim, self.hidden_dim, concat=args.skipConcat, compress=args.skipCompress)

        self.hidden_dim *= 2
        self.fc1 = nn.Linear(self.hidden_dim + args.ex_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, args.out_dim)

    def forward(self, g):
        # Conv Layer 1
        id1 = g.x
        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = self.skip1(id1, h1)

        # Conv Layer 2
        id2 = h1
        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = F.elu(h2)
        h2 = self.skip2(id2, h2)


        # Conv Layer 3
        id3 = h2
        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = F.elu(h3)
        h3 = self.skip3(id3, h3)

        h3 = self.redconv1(h3)
        # h3 = self.bottlenorm(h3)
        h3 = F.elu(h3)

        # Conv Layer 4
        id4 = h3
        # h4 = self.conv4(torch.cat([h3, id1], dim=1), g.edge_index)
        h4 = self.conv4(h3, g.edge_index)
        h4 = self.norm4(h4)
        h4 = F.elu(h4)
        h4 = self.skip4(id4, h4)

        # Conv Layer 5
        id5 = h4
        h5 = self.conv5(h4, g.edge_index)
        h5 = self.norm5(h5)
        h5 = F.elu(h5)
        h5 = self.skip5(id5, h5)

        # Conv Layer 6
        id6 = h5
        h6 = self.conv6(h5, g.edge_index)
        h6 = self.norm6(h6)
        h6 = F.elu(h6)
        h6 = self.skip6(id6, h6)


        # Conv Layer 7
        id7 = h6
        h7 = self.conv7(h6, g.edge_index)
        h7 = self.norm7(h7)
        h7 = F.elu(h7)
        h7 = self.skip7(id7, h7)

        # id8 = h7
        # h8 = self.conv8(h7, g.edge_index)
        # h8 = self.norm8(h8)
        # h8 = F.elu(h8)
        # h8 = self.skip8(id8, h8)


        # Read Out Layer 
        hg = geonn.global_mean_pool(h7, g.batch)

        # extend Strcture(molecule) Feature
        hg = torch.cat([hg, g.eFeature],
                       dim=1)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.elu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out



class FNN(nn.Module):
    def __init__(self, args):
        super(FNN, self).__init__()

        self.hidden_dim = args.hidden_dim
        
        # self.hidden_dim *= 2
        self.fc1 = nn.Linear(args.input_dim, self.hidden_dim)
        self.norm1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, args.out_dim)

    def forward(self, g):
        # extend Strcture(molecule) Feature
        # hg = g.eFeature
        hg = g.ecfp

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = self.norm1(hg)
        hg = F.relu(hg)

        # FCN Layer 2
        hg = self.fc2(hg)
        hg = self.norm2(hg)
        hg = F.relu(hg)

        # FCN Layer 3 
        hg = self.fc3(hg)
        hg = self.norm3(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc4(hg)

        return out




class KAIST_GAT(nn.Module):
    def __init__(self, args):
        super(KAIST_GAT, self).__init__()

        args.head_num = 4  # KAIST 논문에서는 K를 4로, summation으로 수행함
        
        self.conv1 = geonn.GATConv(args.input_dim, args.hidden_dim, heads=args.head_num, concat=False)
        self.conv2 = geonn.GATConv(args.hidden_dim, args.hidden_dim, heads=args.head_num, concat=False)
        self.conv3 = geonn.GATConv(args.hidden_dim, args.hidden_dim, heads=args.head_num, concat=False)

        self.skip1 = GatedSkipConnection(args.input_dim, args.hidden_dim)
        self.skip2 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)
        self.skip3 = GatedSkipConnection(args.hidden_dim, args.hidden_dim)

        self.norm1 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)
        self.norm2 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)
        self.norm3 = geonn.BatchNorm(args.hidden_dim, args.hidden_dim)

        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.out_dim)

    def forward(self, g):
         # Conv Layer 1
        id1 = g.x
        h1 = self.conv1(g.x, g.edge_index)
        h1 = self.norm1(h1)
        h1 = self.skip1(id1, h1)
        h1 = F.relu(h1)


        # Conv Layer 2
        id2 = h1
        h2 = self.conv2(h1, g.edge_index)
        h2 = self.norm2(h2)
        h2 = self.skip2(id2, h2)
        h2 = F.relu(h2)


        # Conv Layer 3
        id3 = h2
        h3 = self.conv3(h2, g.edge_index)
        h3 = self.norm3(h3)
        h3 = self.skip3(id3, h3)
        h3 = F.relu(h3)
        

        # Read Out Layer 
        hg = geonn.global_mean_pool(h3, g.batch)

        # FCN Layer 1 
        hg = self.fc1(hg)
        hg = F.relu(hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
