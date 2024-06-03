import torch
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import show_video

class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = gym_env
        self.state_processing_function = obs_processing_func # Funcion phi para procesar los estados.
        # self.memory = memory_buffer_size # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size, device) # Asignarle memoria al agente
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time # no lo uso
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block
        self.total_steps = 0
    
    
    # def train(self, number_episodes = 50000, max_steps_episode = 10000, max_steps = 1_000_000, writer_name="default_writer_name"):
    def train(self, number_episodes, max_steps_episode, max_steps, writer_name="default_writer_name"):
        
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment="-" + writer_name)
  
        for ep in tqdm(range(number_episodes), unit=' episodes'):
            if total_steps > max_steps:
                break # si supero la maxima cantidad de iteraciones, rompe el bucle
        
        # Observar estado inicial como indica el algoritmo
        state, info = self.env.reset()   # inicializo state: Tiene 4 frames de 84x84 ??????????????????????????? OK ??????????????
        frames = len(state)
        height, width = len(state[0][0]), len(state[0][0])
        n_actions = self.env.action_space.n
        q_table = np.zeros( (frames, height, width, n_actions) ) # 4*84*84*6 --> frames * height * width * acion_space
        current_episode_reward = 0.0
        
        for s in range(max_steps):
            
            # Seleccionar accion usando una política epsilon-greedy.
            epsilon_update = self.compute_epsilon(s)
            rnd = np.random.uniform()
            if rnd < epsilon_update:
                action = np.random.choice(self.env.action_space.n) # exploracion
            else:
                action = np.argmax(q_table[frames][height][width]) # explotacion ??????????????????????????? Q_TABLE ????????????????????????????????
              
            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            self.env.reset()
            next_state, reward, done, truncated, info = self.env.step(action)
            current_episode_reward += reward
            total_steps += 1
  
            # Guardar la transicion en la memoria
            experience = Transition(state, action, reward, done, next_state)
            self.memory.add(experience)

            # Actualizar el estado


            # Actualizar el modelo
  
            if done or truncated: 
                break
          
            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)
  
            # Report on the traning rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                avg_reward_last_eps = np.mean(rewards[-self.episode_block:])
                print(f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {avg_reward_last_eps} epsilon {self.epsilon} total steps {total_steps}")
  
        print(f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")
  
        torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
        writer.close()
  
        return rewards
    
        
    def compute_epsilon(self, steps_so_far):
         new_epsilon = self.epsilon_i - (steps_so_far * self.epsilon_decay)
         new_epsilon = max(new_epsilon, self.epsilon_f)
         return new_epsilon
    
    
    def record_test_episode(self, env):
        done = False
    
        # Observar estado inicial como indica el algoritmo 
        
        env.start_video_recorder()
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.

            if done:
                break      

            # Actualizar el estado  
        env.close_video_recorder()
        env.close()
        show_video()

    ################################ SE MODIFICAN EN CADA AGENTE ###################################
    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass
    #################################################################################################
