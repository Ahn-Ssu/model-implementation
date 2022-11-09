"""
Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
arXiv preprint arXiv:1609.02907. https://arxiv.org/abs/1609.02907 
"""
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