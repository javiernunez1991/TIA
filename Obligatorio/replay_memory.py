import random
from collections import namedtuple, deque
# import numpy as np
# import numpy as np
# import torch

Transition = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])

class ReplayMemory:

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size) # objeto similar a lista pero de largo fijo
    
    def add(self, state, action, reward, done, next_state):
        e = Transition(state, action, reward, done, next_state) 
        self.memory.append(e) # Agrega una nueva experiencia a la memoria

    def sample(self, batch_size):
        batch = random.sample(self.memory, k=batch_size) # obtiene una muestra aleatoria de tamaño batch_size
        return batch

    def __len__(self):
        return len(self.memory) # Tamaño actual de la memoria


## CHEQUEO SI FUNCIONA  
# buffer_size = 7
# batch_size = 3
# replay = ReplayMemory(buffer_size)
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 0, 111, True, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 5, 222, False, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 4, 333, True, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE)) 
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 2, 444, True, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 1, 555, False, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 0, 666, True, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 3, 777, False, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 3, 888, False, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 4, 999, True, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# replay.add(process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE), 2, 101010, False, process_state(np.random.rand(4, 84, 84).astype(np.float32), DEVICE))
# mini_batch = replay.sample(batch_size)
# states = torch.stack([mini_batch[i].state for i in range(batch_size)])
# actions = torch.tensor([mini_batch[i].action for i in range(batch_size)])
# rewards = torch.tensor([mini_batch[i].reward for i in range(batch_size)])
# dones = torch.tensor([int(mini_batch[i].done) for i in range(batch_size)])
# next_states = torch.stack([mini_batch[i].next_state for i in range(batch_size)])

