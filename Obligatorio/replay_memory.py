import random
from collections import namedtuple, deque

Transition = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])

class ReplayMemory:

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size) # objeto similar a lista pero de largo fijo
    
    def add(self, e):
        # e = Transition(state, action, reward, done, next_state) 
        self.memory.append(e) # Agrega una nueva experiencia a la memoria

    def sample(self, batch_size):
        batch = random.sample(self.memory, k=batch_size) # obtiene una muestra aleatoria de tamaño batch_size
        return batch

    def __len__(self):
        return len(self.memory) # Tamaño actual de la memoria
