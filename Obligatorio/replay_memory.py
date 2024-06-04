import random
from collections import namedtuple, deque
# import numpy as np
# import torch

Transition = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])

class ReplayMemory:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size # numero maximo de experiences que almacena la replay memory        
        self.memory = deque(maxlen=buffer_size) # objeto similar a lista pero de largo fijo
        self.experience = Transition('state', 'action', 'reward', 'done', 'next_state')
    
    def add(self, state, action, reward, done, next_state):
        e = self.experience(state, action, reward, done, next_state) 
        self.memory.append(e) # Agrega una nueva experiencia a la memoria

    def sample(self, batch_size):
        batch = random.sample(self.memory, k=self.batch_size) # obtiene una muestra aleatoria de tamaño batch_size
        return batch

    def __len__(self):
        return len(self.memory) # Tamaño actual de la memoria
