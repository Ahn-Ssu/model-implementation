import torch
import torch.nn as nn

class GatedSkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim, concat=False, compress=False):
        super(GatedSkipConnection, self).__init__()
        if compress:
            concat = True

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat
        self.compress = compress

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

        if self.compress:
            self.comp_in = nn.Linear(out_dim, int(out_dim/2), bias=False)
            self.comp_out = nn.Linear(out_dim, int(out_dim/2), bias=False)
            self.linear_coef_in = nn.Linear(int(out_dim/2), int(out_dim/2))
            self.linear_coef_out = nn.Linear(int(out_dim/2), int(out_dim/2))

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)


        if self.compress:
            in_x = self.comp_in(in_x)
            out_x = self.comp_out(out_x)

        z = self.gate_coefficient(in_x, out_x)
        
        if self.concat:
            out = torch.cat(
                (torch.mul(z, out_x), torch.mul(1.0-z, in_x)),
                dim=1)
        else:
            out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out
            
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)
        
class SkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim, concat=False, compress=False):
        super(SkipConnection, self).__init__()

        if compress:
            concat = True
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat
        self.compress = compress
        
        if (self.in_dim != self.out_dim):
            self.linear = nn.Linear(in_dim, out_dim, bias=False)

        if self.compress:
            self.comp_in = nn.Linear(out_dim, int(out_dim/2), bias=False)
            self.comp_out = nn.Linear(out_dim, int(out_dim/2), bias=False)
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)

        if self.compress:
            in_x = self.comp_in(in_x)
            out_x = self.comp_out(out_x)

        if self.concat:
            out = torch.cat((in_x,out_x), dim=1)
        else:
            out = in_x + out_x
        return out
