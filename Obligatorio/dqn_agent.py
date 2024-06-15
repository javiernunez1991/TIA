import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from replay_memory import ReplayMemory, Transition
from abstract_agent import Agent
# from tqdm.notebook import tqdm
import numpy as np



class DQNAgent(Agent):
    
    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, 
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device):
       
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, 
                         epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device)
        
        self.policy_net = model.to(device) # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Asignar un optimizador (Adam)
        
                 
    # @abstractmethod de abstract_agent.py
    def select_action(self, state, train=True):
        
        # Seleccionando acciones epsilongreedy-mente si estamos entrenando y completamente greedy en otro caso.
        if train:
            rnd = np.random.uniform()
            if rnd < self.epsilon_i:
                action = np.random.choice(self.env.action_space.n) # exploracion
            else:
                aux = state.unsqueeze(0)
                q_values = self.policy_net(aux)
                action = np.argmax(q_values.tolist()[0]) # explotacion
        else:
            aux = state.unsqueeze(0)
            q_values = self.policy_net(aux)
            action = np.argmax(q_values.tolist()[0]) # explotacion
    
        return action


    # @abstractmethod de abstract_agent.py
    def update_weights(self):
        if len(self.memory) > self.batch_size:
          
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 
            mini_batch = self.memory.sample(self.batch_size)
            states = torch.stack([mini_batch[i].state for i in range(self.batch_size)])#.to(self.device)
            actions = torch.tensor([mini_batch[i].action for i in range(self.batch_size)]).to(self.device) 
            rewards = torch.tensor([mini_batch[i].reward for i in range(self.batch_size)]).to(self.device)
            dones = torch.tensor([int(mini_batch[i].done) for i in range(self.batch_size)]).to(self.device) # paso los T-F, a 1-0
            next_states = torch.stack([mini_batch[i].next_state for i in range(self.batch_size)])#.to(self.device)

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_values = self.policy_net(states) # states es un minibatch de: (batch_size x 4 x 84 x 84)
            state_q_values = q_values.gather(1, actions.unsqueeze(1))#.squeeze()#.cpu().sum().item() #ACTIONS VECTOR VERTICAL 
            

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            next_q_values = self.policy_net(next_states)
            q_next_states = torch.max(next_q_values, dim=1)#.cpu().sum().item()
            max_next_q_values = q_next_states.values.detach()
        

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
            target = rewards + (1 - dones) * self.gamma * max_next_q_values


            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            loss = F.mse_loss(target, state_q_values) # Calculate el error
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Update model weights
