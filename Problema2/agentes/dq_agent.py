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
        # Acceder directamente a las propiedades del juego
        self.pipe_gap = self.game.pipe_gap
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
            'player_relative_y_position': 3,   # Informa a qué altura está volando el pajaro según la ubicación de las tuberías más próximas
            'player_velocity_sign': 3, # Informa la dirección del vuelo del pájaro
            'player_danger': 9, # Informa el grado de peligrosidad del juego al combinar player_relative_y_position y player_velocity_sign
            'next_pipe_distance': 3, # Informa grado de cercanía del pájaro a las tuberías más próximas
            'next_next_pipe_relative_position': 3, # Informa la posición de las tuberías más alejadas en relación a las tuberías más próximas
        }
        # Variables auxiliares para la discretización
        self.player_velocity_threshold = 5 
        self.next_pipe_distance_threshold_far = 137 
        self.next_pipe_distance_threshold_near = 73
        self.next_next_pipe_relative_position_threshold = 30

    def discretize_state(self, state):
        """
        Permite pasar de un estado continuo a otro discreto customerizado
        """
        # player_relative_y_position_bin
        if state['player_y'] < state['next_pipe_top_y']:
            player_relative_y_position_bin = 0 # Arriba del gap
        elif state['player_y'] > state['next_pipe_bottom_y']:
            player_relative_y_position_bin = 2 # Abajo del gap   
        else:
            player_relative_y_position_bin = 1 # A la altura del gap          
        
        # player_velocity_sign
        if state['player_vel'] <= -self.player_velocity_threshold:
            player_velocity_sign_bin = 0 # Vuelo ascendente    
        elif state['player_vel'] >= self.player_velocity_threshold:
            player_velocity_sign_bin = 2 # Vuelo descendente
        else:
            player_velocity_sign_bin = 1 # Vuelo plano

        # player_danger
        if player_relative_y_position_bin == 1 and player_velocity_sign_bin == 1:
            player_danger_bin = 0 # A la altura del gap y volando plano (muy bajo peligro)
        elif player_relative_y_position_bin == 0 and player_velocity_sign_bin == 2:
            player_danger_bin = 1 # Arriba del gap y bajando (bajo peligro)
        elif player_relative_y_position_bin == 2 and player_velocity_sign_bin == 0:
            player_danger_bin = 2 # Debajo del gap y subiendo (bajo peligro)
        elif player_relative_y_position_bin == 1 and player_velocity_sign_bin == 0:
            player_danger_bin = 3 # En el gap y subiendo (peligro)
        elif player_relative_y_position_bin == 1 and player_velocity_sign_bin == 2:
            player_danger_bin = 4 # En el gap y bajando (peligro)
        elif player_relative_y_position_bin == 0 and player_velocity_sign_bin == 1:
            player_danger_bin = 5 # Arriba del gap y volando plano (alto peligro)
        elif player_relative_y_position_bin == 2 and player_velocity_sign_bin == 1:
            player_danger_bin = 6 # Debajo del gap y volando plano (alto peligro)
        elif player_relative_y_position_bin == 0 and player_velocity_sign_bin == 0:
            player_danger_bin = 7 # Arriba del gap y subiendo (peligro extremo)
        else: #player_relative_y_position_bin == 2 and player_velocity_sign_bin == 2:
            player_danger_bin = 8 # Debajo del gap y bajando (peligro extremo)

        # next_pipe_distance
        next_pipe_distance_bin = None
        if state['next_pipe_dist_to_player'] < self.next_pipe_distance_threshold_near:
            next_pipe_distance_bin = 0 # Cerca    
        elif state['next_pipe_dist_to_player'] < self.next_pipe_distance_threshold_far:
            next_pipe_distance_bin = 1 # Distante
        else:
            next_pipe_distance_bin = 2 # Muy Distante
        # next_next_pipe_relative_position 
        if (state['next_next_pipe_top_y'] < ( state['next_pipe_top_y'] - self.next_next_pipe_relative_position_threshold) ):
            next_next_pipe_relative_position_bin = 0 # El gap de las tuberías más alejadas está arriba del gap de las tuberías más próximas
        elif (state['next_next_pipe_top_y'] > ( state['next_pipe_top_y'] + self.next_next_pipe_relative_position_threshold) ):
            next_next_pipe_relative_position_bin = 2 # El gap de las tuberías más alejadas está debajo del gap de las tuberías más próximas
        else:
            next_next_pipe_relative_position_bin = 0 # El gap de las tuberías más alejadas está alineado con el gap de las tuberías más próximas

        return (
            player_relative_y_position_bin,
            player_velocity_sign_bin,
            player_danger_bin,
            next_pipe_distance_bin,
            next_next_pipe_relative_position_bin,
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
        print('estado discretizado',discrete_state)
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
