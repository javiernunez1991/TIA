import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory, Transition
from abstract_agent import Agent
from tqdm.notebook import tqdm
import numpy as np



class DQNAgent(Agent):
    
    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, 
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device):
       
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, 
                         epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device)
        
        self.policy_net = model.to(device) # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.loss_function = nn.CrossEntropyLoss().to(device) # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9) # Asignar un optimizador (Adam)
        
        
    # @abstractmethod de abstract_agent.py
    def select_action(self, state, current_steps, train=True):
      # Implementar. Seleccionando acciones epsilongreedy-mente si estamos entranando y completamente greedy en otro caso.
      pass


    # @abstractmethod de abstract_agent.py
    def update_weights(self):
      if len(self.memory) > self.batch_size:
          
            # Resetear gradientes

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 


            # Enviar los tensores al dispositivo correspondiente.
            #states = ?
            #actions = ?
            #rewards = ?
            #dones = ?  # Dones deberia ser 0 y 1; no True y False. Pueden usar .float() en un tensor para convertirlo
            #next_states = ?

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            #q_actual = ?

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            #max_q_next_state = ?

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
            #target = ?

            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            pass
