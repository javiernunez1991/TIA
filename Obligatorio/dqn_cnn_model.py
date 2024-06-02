import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=env_inputs[0], out_channels=16, kernel_size=8, stride=4), # env_inputs[0] = 4 frames
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()          
            )
        
        conv_out_size = self._get_conv_out(env_inputs) # 4x84x84 = 28224 neuronas ocultas
       
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256), # las 28224 fully conected con 256 neuronas
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )        
        
    def _get_conv_out(self, shape):
        #o = self.conv(torch.zeros(1, *shape))
        o = torch.zeros(1, *shape)
        return int(np.prod(o.size()))        


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)        
        
















