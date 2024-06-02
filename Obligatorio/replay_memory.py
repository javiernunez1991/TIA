import random
from collections import namedtuple, deque
import numpy as np
import torch

Transition = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])

class ReplayMemory:

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size # numero maximo de experiences que almacena la replay memory        
        self.memory = deque(maxlen=buffer_size) # objeto similar a lista pero de largo fijo
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])
        self.device = device
    
    def add(self, state, action, reward, done, next_state):
        e = self.experience(state, action, reward, next_state, done) # Agrega una nueva experiencia a la memoria
        self.memory.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory) # Tamaño actual de la memoria