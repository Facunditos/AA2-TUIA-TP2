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
            'relative_pos_y': 25, # 25 valores (a definir)
            'player_velocity': 3,
            'relative_pos_x': 12,
            'distancia_critica_y' : 3
        }
        self.player_v_threshold = 3
        self.tuberia_y_threshold_up = 20
        self.tuberia_y_threshold_bottom = -20


    def discretize_state(self, state):
        """
        Permite pasar de un estado continuo a otro discreto customerizado
        """
        # 1. Posición relativa del jugador al centro del hueco de la tubería
        centro_hueco_tuberia = (state['next_pipe_top_y'] + state['next_pipe_bottom_y'])*0.50  
        pos_rel_centro_hueco_tuberia = centro_hueco_tuberia - state['player_y']
        scales_pos_rel_centro_hueco_tuberia = (pos_rel_centro_hueco_tuberia + self.game.height / 2) / self.game.height
        relative_pos_y_bin = int(np.clip(scales_pos_rel_centro_hueco_tuberia * self.num_bins['relative_pos_y'], 0, self.num_bins['relative_pos_y'] - 1))
        
        # 2. Signo de la velocidad del jugador
        if state['player_vel'] < -self.player_v_threshold:
            player_velocity_sign_bin = 0 # Moviéndose arriba
        elif state['player_vel'] > self.player_v_threshold:
            player_velocity_sign_bin = 2 # Moviéndose abajo
        else:
            player_velocity_sign_bin = 1 # Quieto o casi quieto
        
        # 3. Distancia relativa a la siguiente tubería
        distancia_tuberia_norm =  state['next_pipe_dist_to_player']/self.game.width
        dist_tuberia_bin = int(np.clip(distancia_tuberia_norm * self.num_bins['relative_pos_x'], 0, self.num_bins['relative_pos_x'] - 1))

        # 4 distnaicas criticas entre tuberias

        centro_hueco_tuberia_next = (state['next_next_pipe_top_y'] + state['next_next_pipe_bottom_y']) * 0.50
        pos_rel_centro_hueco_tuberia_next = centro_hueco_tuberia_next - state['player_y']
        if pos_rel_centro_hueco_tuberia_next >= self.tuberia_y_threshold_up:
            relative_critic_y_bin = 0

        elif pos_rel_centro_hueco_tuberia_next < self.tuberia_y_threshold_bottom:
            relative_critic_y_bin = 1
        else:
            relative_critic_y_bin = 2
        return (relative_pos_y_bin, player_velocity_sign_bin, dist_tuberia_bin, relative_critic_y_bin)

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
