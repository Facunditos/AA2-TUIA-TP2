from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random


class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.game = game
        # Acceder directamente a las propiedades del juego
        self.game_pipe_gap = self.game.pipe_gap
        self.game_height = self.game.height # PLE pasa el objeto game directamente
        self.game_width = self.game.width # PLE pasa el objeto game directamente
        # Incorporar la tabla q-table al agente
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.num_bins = {
            'player_velocity_sign': 3, # Informa la dirección del vuelo del pájaro
            'next_gap_relative_y_position': 25,   # Informa la posición central del gap más próximo en relación a la posición del pájaro
            'next_pipe_distance': 10, # Informa grado de cercanía del pájaro a las tuberías más próximas
            'next_next_gap_relative_y_position': 3, # Informa la posición central del gap más alejado en relación a la posición del pájaro
            'next_next_pipe_distance': 5, # Informa grado de cercanía del pájaro a las tuberías más alejadas
        }
        # Variables auxiliares para la discretización
        self.player_velocity_threshold = -4


    def discretize_state(self, state):
        """
        Permite pasar de un estado continuo a otro discreto customerizado
        """
        # player_velocity_sign
        if state['player_vel'] < self.player_velocity_threshold:
            player_velocity_sign_bin = 0 # Vuelo ascendente 
        elif state['player_vel'] > self.player_velocity_threshold:
            player_velocity_sign_bin = 2 # Vuelo descendenteo
        else: # Vuelo descendente muy rápido
            player_velocity_sign_bin = 1 # Vuelo plano

        # next_gap_relative_y_position
        next_gap_center_y = state['next_pipe_top_y'] + self.game_pipe_gap / 2
        relative_next_gap_position_y = next_gap_center_y - state['player_y']
        scaled_relative_next_gap_position_y = (relative_next_gap_position_y + self.game_height / 2) / self.game_height
        relative_next_gap_position_y_bin = int(np.clip(scaled_relative_next_gap_position_y * self.num_bins['next_gap_relative_y_position'], 0, self.num_bins['next_gap_relative_y_position'] - 1))      

        # next_pipe_distance
        next_pipe_distance_bin = None
        next_pipe_distance = state['next_pipe_dist_to_player'] / self.game_width
        next_pipe_distance_bin = int(np.clip(next_pipe_distance * self.num_bins['next_pipe_distance'], 0, self.num_bins['next_pipe_distance'] - 1))

        # next_next_gap_relative_y_position
        if (next_gap_center_y < state['next_next_pipe_top_y']):
            relative_next_next_gap_position_y_bin = 0
        elif (next_gap_center_y<state['next_next_pipe_bottom_y']):
            relative_next_next_gap_position_y_bin = 1
        else:
            relative_next_next_gap_position_y_bin = 2

        # next_next_pipe_distance
        next_next_pipe_distance_bin = None
        next_next_pipe_distance = state['next_next_pipe_dist_to_player'] / 420
        next_next_pipe_distance_bin = int(np.clip(next_next_pipe_distance * self.num_bins['next_next_pipe_distance'], 0, self.num_bins['next_next_pipe_distance'] - 1))           

        return (
            player_velocity_sign_bin,
            relative_next_gap_position_y_bin,
            next_pipe_distance_bin,
            #relative_next_next_gap_position_y_bin,
            #next_next_pipe_distance_bin,
        )

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        """
        # Sugerencia:
        # - Discretizar el estado
        # - Con probabilidad epsilon elegir acción aleatoria
        # - Si no, elegir acción con mayor Q-value
        discrete_state = self.discretize_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
