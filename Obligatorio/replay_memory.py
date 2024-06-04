import random
from collections import namedtuple, deque
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
# replay = ReplayMemory(7)
# replay.add(1, 11, 111, True, 1111)
# replay.add(2, 22, 222, False, 2222)
# replay.add(3, 33, 333, True, 3333) 
# replay.add(4, 44, 444, True, 4444)
# replay.add(5, 55, 555, False, 5555)
# replay.add(6, 66, 666, True, 6666)
# replay.add(7, 77, 777, False, 7777)
# replay.add(8, 88, 888, False, 8888)
# replay.add(9, 99, 999, True, 9999)
# replay.add(10, 1010, 101010, False, 10101010)

# device = 'cpu'
# mini_batch = replay.sample(3)
# states = torch.from_numpy(np.vstack([e.state for e in mini_batch if e is not None])).float().to(device)
# actions = torch.from_numpy(np.vstack([e.action for e in mini_batch if e is not None])).long().to(device)
# rewards = torch.from_numpy(np.vstack([e.reward for e in mini_batch if e is not None])).float().to(device)
# dones = torch.from_numpy(np.vstack([e.done for e in mini_batch if e is not None]).astype(np.uint8)).float().to(device)
# next_states = torch.from_numpy(np.vstack([e.next_state for e in mini_batch if e is not None])).float().to(device)
