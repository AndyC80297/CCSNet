import torch

import torch.nn.functional as F

from torch import nn


class WaveNet(nn.Module):
    
    def __init__(
        self, 
        num_ifos: int,
        c_depth: int=8, 
        n_chann: int=64, 
        l1: int=1024, 
        l2: int=128
    ):
        
        super(WaveNet, self).__init__()
        
        self.c_depth = c_depth
        self.n_chann = n_chann
        
        self.cap_norm = nn.GroupNorm(num_ifos, num_ifos)
        
        self.Conv_In = nn.Conv1d(
                in_channels=num_ifos, 
                out_channels=self.n_chann, 
                kernel_size=2
            )
        
        self.Conv_Out = nn.Conv1d(
                in_channels=self.n_chann, 
                out_channels=1, 
                kernel_size=1
            )
        
        self.body_norm = nn.GroupNorm(4 ,n_chann)
        self.end_norm = nn.BatchNorm1d(1)
        
        self.WaveNet_layers = nn.ModuleList()
        
        
        for i in range(self.c_depth):

            conv_layer = nn.Conv1d(
                in_channels=self.n_chann, 
                out_channels=self.n_chann,
                kernel_size=2,
                dilation=2**i
            )
            
            self.WaveNet_layers.append(conv_layer)
        
        
        # self.L1 = nn.Linear(8192-2**c_depth, l1)
        
        # Consider replacing other batch normalizatoin layers with other nor method
        # Because batch norm are baised by the population of the CCSN rate in one batch 
        # This may produce overfitting model and will not be able to found at test phase
        # Question: Will we be able to figure out the side effect at infereceing phase?
        
        self.L1_norm = nn.BatchNorm1d(4096-2**c_depth)
        self.L1 = nn.Linear(4096-2**c_depth, l1)
        self.L2_norm = nn.BatchNorm1d(l1)
        self.L2 = nn.Linear(l1, l2)
        self.L3_norm = nn.BatchNorm1d(l2)
        self.L3 = nn.Linear(l2, 1)


        nn.init.kaiming_normal_(self.Conv_In.weight)
        nn.init.kaiming_normal_(self.Conv_Out.weight)
        nn.init.constant_(self.Conv_In.bias, 0.001)
        nn.init.constant_(self.Conv_Out.bias, 0.001)

        # Initialize all the convolutional layer in between
        for conv_layer in self.WaveNet_layers:
            nn.init.kaiming_normal_(conv_layer.weight)
            nn.init.constant_(conv_layer.bias, 0.001)

        nn.init.kaiming_uniform_(self.L1.weight)
        nn.init.kaiming_uniform_(self.L2.weight)
        nn.init.constant_(self.L1.bias, 0.001)
        nn.init.constant_(self.L2.bias, 0.001)
        
    def forward(self, x):

        x = self.cap_norm(x)
        x = self.Conv_In(x)
        x = F.relu(x)
        
        # x = self.norm(x)
        
        for what_are_u_wavin_at in self.WaveNet_layers:
            x = self.body_norm(x)
            x = what_are_u_wavin_at(x)
            x = F.relu(x)
            
        x = self.Conv_Out(x)
        x = F.relu(x)
        x = self.end_norm(x)
        
        x = torch.flatten(x, 1)
        
        
        x = F.relu(self.L1(x))
        x = self.L2_norm(x)
        x = F.relu(self.L2(x))
        x = self.L3_norm(x)
        # x = F.softmax(self.L3(x), dim = 1)
        x = self.L3(x)
        
        return x