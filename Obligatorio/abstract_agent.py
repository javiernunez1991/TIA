import torch
from replay_memory import ReplayMemory, Transition
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import show_video

class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device):
        
        self.device = device
        self.env = gym_env
        self.state_processing_function = obs_processing_func # Funcion phi para procesar los estados.
        self.memory = ReplayMemory(memory_buffer_size) # Asignarle memoria al agente
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time # no lo uso
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block
        self.total_steps = 0
        self.epsilon = epsilon_i
        self.epsilon_values = []
        
    
    # def train(self, number_episodes = 50000, max_steps_episode = 10000, max_steps = 1_000_000, writer_name="default_writer_name"):
    def train(self, number_episodes, max_steps_episode, writer_name="default_writer_name"):
        
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment="-" + writer_name)
  
        for ep in tqdm(range(number_episodes), unit=' episodes'):
            if total_steps > max_steps_episode:
                break
        
            # Observar estado inicial como indica el algoritmo
            state, info = self.env.reset()   # inicializo state: Tiene 4 frames de 84x84 
            state = self.state_processing_function(state, self.device) # paso el state a tensor (funcion process_state de la notebook)
            current_episode_reward = 0.0
            done = False
            truncated = False
            
            for s in range(max_steps_episode):
                
                # Seleccionar accion usando una política epsilon-greedy.
                self.epsilon_values.append(self.epsilon)
                self.compute_epsilon()
                action = self.select_action(state, True)
                  
                # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                next_state = self.state_processing_function(next_state, self.device) # paso el state a tensor (funcion process_state de la notebook)
                current_episode_reward += reward
                total_steps += 1
      
                # Guardar la transicion en la memoria
                experience = Transition(state, action, reward, done, next_state)
                self.memory.add(experience)
    
                # Actualizar el estado
                state = next_state
    
                # Actualizar el modelo
                self.update_weights()
        
                if done: 
                    break
              
            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)
            avg_reward_last_eps = np.round(np.mean(rewards[-self.episode_block:]),2)
      
            # Report on the traning rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:    
                print(f"Episode {ep}: Avg. Reward {avg_reward_last_eps} over the last {self.episode_block} episodes - Epsilon {self.epsilon} - TotalSteps {total_steps}")
        print(f"Episode {ep + 1} - Avg. Reward {avg_reward_last_eps} over the last {self.episode_block} episodes - Epsilon {self.epsilon} - TotalSteps {total_steps}")

        
        torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
        writer.close()
  
        return rewards, self.epsilon_values
    
        
    def compute_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_f)
        return self.epsilon
    
    
    def record_test_episode(self, env):
        done = False
    
        # Observar estado inicial como indica el algoritmo
        state, info = env.reset()
        env.start_video_recorder()
        
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(self.state_processing_function(state, self.device), False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            next_state, reward, done, truncated, info = env.step(action)

            if done or truncated:
                break      

            # Actualizar el estado
            state = next_state
            
        env.close_video_recorder()
        env.close()
        show_video()
        
        del env


    ################################ SE MODIFICAN EN CADA AGENTE ###################################
    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass
    #################################################################################################
