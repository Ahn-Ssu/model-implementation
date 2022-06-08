import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geonn
from skipNetwork import GatedSkipConnection, SkipConnection


class InceptlikeGCN(nn.Module):
    def __init__(self, args):
        super(InceptlikeGCN, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.head_num = args.head_num 
        self.ex_dim = args.ex_dim
        
        self.conv11 = geonn.GCNConv(self.input_dim, self.hidden_dim)
        self.conv12 = geonn.GATConv(self.input_dim, int(self.hidden_dim/self.head_num),heads=self.head_num, negative_slope=0.2)
        self.conv13 = geonn.LEConv(self.input_dim, self.hidden_dim)
        self.conv14 = geonn.SAGEConv(self.input_dim, self.hidden_dim)
        self.conv15 = geonn.ClusterGCNConv(self.input_dim, self.hidden_dim)

        
        self.conv21 = geonn.GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv31 = geonn.GCNConv(self.hidden_dim, self.hidden_dim)
        # Attention
        
        self.conv22 = geonn.GATConv(self.hidden_dim, int(self.hidden_dim/self.head_num),heads=self.head_num, negative_slope=0.2)
        self.conv32 = geonn.GATConv(self.hidden_dim, int(self.hidden_dim/self.head_num),heads=self.head_num, negative_slope=0.2)
        # LE 
        
        self.conv23 = geonn.LEConv(self.hidden_dim, self.hidden_dim)
        self.conv33 = geonn.LEConv(self.hidden_dim, self.hidden_dim)
        # SAGE
        
        self.conv24 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv34 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)
        # Clsuter
        
        self.conv25 = geonn.ClusterGCNConv(self.hidden_dim, self.hidden_dim)
        self.conv35 = geonn.ClusterGCNConv(self.hidden_dim, self.hidden_dim)
        
        self.skip1 = GatedSkipConnection(self.input_dim, self.hidden_dim)
        self.skip2 = GatedSkipConnection(self.hidden_dim, self.hidden_dim)
        self.skip3 = GatedSkipConnection(self.hidden_dim, self.hidden_dim)
        # self.skip4 = SkipConnection(self.hidden_dim+20, self.hidden_dim)
        # self.skip5 = SkipConnection(self.hidden_dim, self.hidden_dim*2)
        # self.gs_inception = GatedSkipConnection

        self.norm11 = geonn.BatchNorm(self.hidden_dim)
        self.norm21 = geonn.BatchNorm(self.hidden_dim)
        self.norm31 = geonn.BatchNorm(self.hidden_dim)

        self.norm12 = geonn.BatchNorm(self.hidden_dim)
        self.norm22 = geonn.BatchNorm(self.hidden_dim)
        self.norm32 = geonn.BatchNorm(self.hidden_dim)

        self.norm13 = geonn.BatchNorm(self.hidden_dim)
        self.norm23 = geonn.BatchNorm(self.hidden_dim)
        self.norm33 = geonn.BatchNorm(self.hidden_dim)

        self.norm14 = geonn.BatchNorm(self.hidden_dim)
        self.norm24 = geonn.BatchNorm(self.hidden_dim)
        self.norm34 = geonn.BatchNorm(self.hidden_dim)

        self.norm15 = geonn.BatchNorm(self.hidden_dim)
        self.norm25 = geonn.BatchNorm(self.hidden_dim)
        self.norm35 = geonn.BatchNorm(self.hidden_dim)
        # self.norm4 = geonn.BatchNorm(self.hidden_dim)
        # self.norm5 = geonn.BatchNorm(self.hidden_dim*2)

        self.fc1 = nn.Linear(self.hidden_dim+20, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, g):
        # Conv Layer 1
        id1 = g.x
        h11 = self.conv11(g.x, g.edge_index)
        h11 = self.norm11(h11)
        h11 = F.relu(h11)

        h12 = self.conv12(g.x, g.edge_index)
        h12 = self.norm12(h12)
        h12 = F.relu(h12)

        h13 = self.conv13(g.x, g.edge_index)
        h13 = self.norm13(h13)
        h13 = F.relu(h13)
        
        h14 = self.conv14(g.x, g.edge_index)
        h14 = self.norm14(h14)
        h14 = F.relu(h14)

        h15 = self.conv15(g.x, g.edge_index)
        h15 = self.norm15(h15)
        h15 = F.relu(h15)

        h1 = h11 + h12 + h13 + h14 + h15
        h1 = self.skip1(id1, h1)

        id2 = h1
        h21 = self.conv21(h1, g.edge_index)
        h21 = self.norm21(h21)
        h21 = F.relu(h21)

        h22 = self.conv22(h1, g.edge_index)
        h22 = self.norm22(h22)
        h22 = F.relu(h22)

        h23 = self.conv23(h1, g.edge_index)
        h23 = self.norm23(h23)
        h23 = F.relu(h23)
        
        h24 = self.conv24(h1, g.edge_index)
        h24 = self.norm24(h24)
        h24 = F.relu(h24)

        h25 = self.conv25(h1, g.edge_index)
        h25 = self.norm25(h25)
        h25 = F.relu(h25)

        h2 = h21 + h22 + h23 + h24 + h25
        h2 = self.skip2(id2, h2)

        id3 = h2
        h31 = self.conv31(h2, g.edge_index)
        h31 = self.norm31(h31)
        h31 = F.relu(h31)

        h32 = self.conv32(h2, g.edge_index)
        h32 = self.norm32(h32)
        h32 = F.relu(h32)

        h33 = self.conv33(h2, g.edge_index)
        h33 = self.norm33(h33)
        h33 = F.relu(h33)
        
        h34 = self.conv34(h2, g.edge_index)
        h34 = self.norm34(h34)
        h34 = F.relu(h34)

        h35 = self.conv35(h2, g.edge_index)
        h35 = self.norm35(h35)
        h35 = F.relu(h35)

        h3 = h31 + h32 + h33 + h34 + h35
        h3 = self.skip3(id3, h3)

        # Read Out Layer 
        hg = geonn.global_mean_pool(h3, g.batch)

        # eGCN, concat Layer
        hg = torch.cat([hg, g.eFeature],
                       dim=1)
        
        # FCN Layer 1 
        # fcId1 = hg
        hg = self.fc1(hg)
        # hg = self.norm4(hg)
        hg = F.relu(hg)
        # hg = self.skip4(fcId1, hg)

        # fcId2 = hg
        # hg = self.fc2(hg)
        # hg = self.norm5(hg)
        # hg = F.relu(hg)
        # hg = self.skip5(fcId2, hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out

class LW_inceptLikeGCN(nn.Module):
    def __init__(self, args):
        super(LW_inceptLikeGCN, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.head_num = args.head_num 
        self.ex_dim = args.ex_dim
        
        self.conv11 = geonn.TAGConv(self.input_dim, self.hidden_dim)
        self.conv12 = geonn.LEConv(self.input_dim, self.hidden_dim)
        self.conv13 = geonn.SAGEConv(self.input_dim, self.hidden_dim)

        self.conv21 = geonn.TAGConv(self.hidden_dim, self.hidden_dim)
        self.conv22 = geonn.LEConv(self.hidden_dim, self.hidden_dim)
        self.conv23 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)

        self.conv31 = geonn.TAGConv(self.hidden_dim, self.hidden_dim)
        self.conv32 = geonn.LEConv(self.hidden_dim, self.hidden_dim)
        self.conv33 = geonn.SAGEConv(self.hidden_dim, self.hidden_dim)
        
        self.skip1 = GatedSkipConnection(self.input_dim, self.hidden_dim)
        self.skip2 = GatedSkipConnection(self.hidden_dim, self.hidden_dim)
        self.skip3 = GatedSkipConnection(self.hidden_dim, self.hidden_dim)
        # self.skip4 = SkipConnection(self.hidden_dim+20, self.hidden_dim)
        # self.skip5 = SkipConnection(self.hidden_dim, self.hidden_dim*2)
        # self.gs_inception = GatedSkipConnection

        self.norm11 = geonn.BatchNorm(self.hidden_dim)
        self.norm21 = geonn.BatchNorm(self.hidden_dim)
        self.norm31 = geonn.BatchNorm(self.hidden_dim)

        self.norm12 = geonn.BatchNorm(self.hidden_dim)
        self.norm22 = geonn.BatchNorm(self.hidden_dim)
        self.norm32 = geonn.BatchNorm(self.hidden_dim)

        self.norm13 = geonn.BatchNorm(self.hidden_dim)
        self.norm23 = geonn.BatchNorm(self.hidden_dim)
        self.norm33 = geonn.BatchNorm(self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim+20, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, g):
        # Conv Layer 1
        id1 = g.x
        h11 = self.conv11(g.x, g.edge_index)
        h11 = self.norm11(h11)
        h11 = F.relu(h11)

        h12 = self.conv12(g.x, g.edge_index)
        h12 = self.norm12(h12)
        h12 = F.relu(h12)

        h13 = self.conv13(g.x, g.edge_index)
        h13 = self.norm13(h13)
        h13 = F.relu(h13)
  

        h1 = h11 + h12 + h13
        h1 = self.skip1(id1, h1)

        id2 = h1
        h21 = self.conv21(h1, g.edge_index)
        h21 = self.norm21(h21)
        h21 = F.relu(h21)

        h22 = self.conv22(h1, g.edge_index)
        h22 = self.norm22(h22)
        h22 = F.relu(h22)

        h23 = self.conv23(h1, g.edge_index)
        h23 = self.norm23(h23)
        h23 = F.relu(h23)
        

        h2 = h21 + h22 + h23 
        h2 = self.skip2(id2, h2)

        id3 = h2
        h31 = self.conv31(h2, g.edge_index)
        h31 = self.norm31(h31)
        h31 = F.relu(h31)

        h32 = self.conv32(h2, g.edge_index)
        h32 = self.norm32(h32)
        h32 = F.relu(h32)

        h33 = self.conv33(h2, g.edge_index)
        h33 = self.norm33(h33)
        h33 = F.relu(h33)
        

        h3 = h31 + h32 + h33
        h3 = self.skip3(id3, h3)

        # Read Out Layer 
        hg = geonn.global_mean_pool(h3, g.batch)

        # eGCN, concat Layer
        hg = torch.cat([hg, g.eFeature],
                       dim=1)
        
        # FCN Layer 1 
        # fcId1 = hg
        hg = self.fc1(hg)
        # hg = self.norm4(hg)
        hg = F.relu(hg)
        # hg = self.skip4(fcId1, hg)

        # fcId2 = hg
        # hg = self.fc2(hg)
        # hg = self.norm5(hg)
        # hg = F.relu(hg)
        # hg = self.skip5(fcId2, hg)

        # FCN Layer out
        out = self.fc3(hg)

        return out
