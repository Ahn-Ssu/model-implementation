import sys
sys.path.append("/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/")

import torch
import torch.nn as nn
import torch_geometric.nn as geonn


class FCN(nn.Module):
    def __init__(self, args):
        super(FCN, self).__init__()

        self.hidden_dim = args.hidden_dim 


        self.fcn = nn.Sequential(
            # Layer 1 
            nn.Linear(args.input_dim+args.ex_dim, self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            
            # Layer 2 
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.ReLU(),

            # Layer 3 
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),

            # Layer 4 
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),

            # Layer 5 ; for out
            nn.Linear(self.hidden_dim, args.out_dim),
        )

    def forward(self, input):

        hg = geonn.global_add_pool(input.x, input.batch)
        hg = torch.cat([hg, input.eFeature],
                           dim=1)

        return self.fcn(hg) 


class NIN2d(nn.Module):
    def __init__(self, args):
        super(NIN2d, self).__init__()

        self.nin = nn.Sequential(
            nn.Conv2d(in_channels=args.hidden_dim + args.ex_dim,
                             out_channels=args.hidden_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=args.hidden_dim,
                             out_channels=args.hidden_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=args.hidden_dim,
                             out_channels=args.out_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0),
        )
    def forward(self, input):
        ex1= torch.unsqueeze(input, -1)
        ex2= torch.unsqueeze(ex1, -1)

        out = self.nin(ex2)

        out = out.view(out.size(0), -1)

        return out

        

class NIN1d(nn.Module):
    def __init__(self, args):
        super(NIN1d, self).__init__()

        self.nin = nn.Sequential(
            nn.Conv1d(in_channels=args.hidden_dim + args.ex_dim,
                             out_channels=args.hidden_dim,
                             kernel_size=3,
                             stride=1,
                             padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=args.hidden_dim,
                             out_channels=args.hidden_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0),
            nn.ReLU(),

            nn.Conv1d(in_channels=args.hidden_dim,
                             out_channels=args.out_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0),
        )
    def forward(self, input):
        ex1= torch.unsqueeze(input, -1)

        out = self.nin(ex1)

        out = out.view(out.size(0), -1)

        return out