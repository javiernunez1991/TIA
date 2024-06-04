import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
        self.loss_function = nn.CrossEntropyLoss().to(device) # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9) # Asignar un optimizador (Adam)
        
                 
    # @abstractmethod de abstract_agent.py
    def select_action(self, state, current_steps, train=True):
        
        # Seleccionando acciones epsilongreedy-mente si estamos entranando y completamente greedy en otro caso.
        if train:
            epsilon_update = self.compute_epsilon(current_steps)
            rnd = np.random.uniform()
            if rnd < epsilon_update:
                action = np.random.choice(self.env.action_space.n) # exploracion
            else:
                with torch.no_grad():
                    aux = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    # aux = state.unsqueeze(0).to(self.device)
                    q_values = self.policy_net(aux)
                    action = np.argmax(q_values.tolist()[0]) # explotacion
        else:
            with torch.no_grad():
                aux = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # aux = state.unsqueeze(0).to(self.device)
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
            # states = torch.from_numpy(np.vstack([e.state for e in mini_batch if e is not None])).float().to(self.device)
            # actions = torch.from_numpy(np.vstack([e.action for e in mini_batch if e is not None])).long().to(self.device)
            # rewards = torch.from_numpy(np.vstack([e.reward for e in mini_batch if e is not None])).float().to(self.device)
            # dones = torch.from_numpy(np.vstack([e.done for e in mini_batch if e is not None]).astype(np.uint8)).float().to(self.device)
            # next_states = torch.from_numpy(np.vstack([e.next_state for e in mini_batch if e is not None])).float().to(self.device)
            states = torch.stack([mini_batch[i].state for i in range(self.batch_size)])
            actions = torch.tensor([mini_batch[i].action for i in range(self.batch_size)])
            rewards = torch.tensor([mini_batch[i].reward for i in range(self.batch_size)])
            dones = torch.tensor([int(mini_batch[i].done) for i in range(self.batch_size)]) # paso los T-F, a 1-0
            next_state = torch.stack([mini_batch[i].next_state for i in range(self.batch_size)])


            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = self.policy_net(states)

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            #max_q_next_state = ?

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
            #target = ?

            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            
            pass
