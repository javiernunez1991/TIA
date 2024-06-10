import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4), # env_inputs[0] = 4 frames
            nn.ReLU(), # aplico reLU sobre los kernels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU() # aplico reLU sobre los kernels         
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_out(state_shape), 256), # conecto las neuronas obtenidas con 256 nuevas neuronas ocultas
            nn.ReLU(), # aplico reLU
            nn.Linear(256, n_actions) # conecto las 256 neuronas ocultas, con la capa final de 6 neuronas (tantas como acciones posibles)
        )        
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape)) 
          # Creo un tensor de 0s, con el mismo shape que el parametro env_inputs
          # Le aplico las mismas capas de convolución que le aplicaria a las imagenes
        o = int( np.prod(o.size()) ) # obtengo la cantidad final de neuronas que necesito
        return o 


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1) # Aplano los feature maps
        q_values = F.softmax(self.fc(conv_out), dim=1)
        return q_values       
