import random
from collections import namedtuple, deque
import numpy as np
# import numpy as np
# import torch

def random_exper(dev):
  return process_state(np.random.rand(4, 84, 84).astype(np.float32), dev)

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
        # states, actions, rewards, dones, next_states = random.sample(self.memory, k=batch_size)
        # return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory) # Tamaño actual de la memoria
# buffer_size = 7
# batch_size = 3
# replay = ReplayMemory(buffer_size)
# e = Transition(random_exper(DEVICE), 0, 111, True, random_exper(DEVICE))
# replay.add(e)
# replay.add(Transition(random_exper(DEVICE), 0, 111, True, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 5, 222, False, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 4, 333, True, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 2, 444, True, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 1, 555, False, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 0, 666, True, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 3, 777, False, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 3, 888, False, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 4, 999, True, random_exper(DEVICE)))
# replay.add(Transition(random_exper(DEVICE), 2, 101010, False, random_exper(DEVICE)))
# mini_batch = replay.sample(batch_size)
# states = torch.stack([mini_batch[i].state for i in range(batch_size)])
# actions = torch.tensor([mini_batch[i].action for i in range(batch_size)])
# rewards = torch.tensor([mini_batch[i].reward for i in range(batch_size)])
# dones = torch.tensor([int(mini_batch[i].done) for i in range(batch_size)])
# next_states = torch.stack([mini_batch[i].next_state for i in range(batch_size)])
