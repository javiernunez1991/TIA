import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from abstract_agent import Agent

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device, sync_target = 100):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device)              
    
        
        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.policy_net = model_a.to(device)
        self.target_net = model_b.to(device)

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = optim.Adam(model_a.parameters(), lr=learning_rate)
        self.optimizer_B = optim.Adam(model_b.parameters(), lr=learning_rate)


    # @abstractmethod de abstract_agent.py
    def select_action(self, state, train=True):
        # Seleccionar acciones epsilongreedy-mente (sobre Q_a + Q_b) si estamos entrenando y completamente greedy en o/c.
        
        if train:
            rnd = np.random.uniform()
            if rnd < self.epsilon:
                action = np.random.choice(self.env.action_space.n) # exploracion
            else:
                aux = state.unsqueeze(0)
                # q_values = self.q_a(aux) + self.q_b(aux)
                q_values = self.policy_net(aux)
                action = np.argmax(q_values.tolist()[0]) # explotacion
        else:
            aux = state.unsqueeze(0)
            # q_values = self.q_a(aux) + self.q_b(aux)
            q_values = self.policy_net(aux)
            action = np.argmax(q_values.tolist()[0]) # explotacion
    
        return action

    
    def update_weights(self):
        if len(self.memory) > self.batch_size:
            if random.random() < 0.5:
                network_to_update = self.policy_net
                optimizer = self.optimizer_A
            else:
                network_to_update = self.target_net
                optimizer = self.optimizer_B
            
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 
            mini_batch = self.memory.sample(self.batch_size)
            states = torch.stack([mini_batch[i].state for i in range(self.batch_size)])#.to(self.device)
            actions = torch.tensor([mini_batch[i].action for i in range(self.batch_size)]).to(self.device) 
            rewards = torch.tensor([mini_batch[i].reward for i in range(self.batch_size)]).to(self.device)
            dones = torch.tensor([int(mini_batch[i].done) for i in range(self.batch_size)]).to(self.device) # paso los T-F, a 1-0
            next_states = torch.stack([mini_batch[i].next_state for i in range(self.batch_size)])#.to(self.device)
            
            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            q_values = network_to_update(states)
            state_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            next_q_values_policy = self.policy_net(next_states).detach()
            next_q_actions = next_q_values_policy.argmax(dim=1).unsqueeze(1)

            next_q_values_target = self.target_net(next_states).detach()
            max_next_q_values = next_q_values_target.gather(1, next_q_actions).squeeze()

            target = rewards + (1 - dones) * self.gamma * max_next_q_values

            loss = F.mse_loss(target.unsqueeze(1), state_q_values)
            loss.backward()
            optimizer.step()

            self.update_count += 1
            if self.update_count % self.sync_target == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
