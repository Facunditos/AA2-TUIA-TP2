from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys
import csv
# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = FlappyBird()  # Usar FlappyBird en vez de Pong
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver

print(game.pipe_gap)
# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet() # [119, None] saltar,no saltar

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True, help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
args = parser.parse_args()
# --- Carga dinámica del agente usando path completo ---
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    print(module_path,class_name)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except (ValueError, ModuleNotFoundError, AttributeError):
    print(f"No se pudo encontrar la clase {args.agent}")
    sys.exit(1)

# Inicializar el agente
agent = AgentClass(actions, game)

# Agente con acciones aleatorias
conjunto_valores = []
atributos = [clave for clave in  {'player_y': 223.0, 'player_vel': 6.0, 'next_pipe_dist_to_player': 241.0, 'next_pipe_top_y': 85, 'next_pipe_bottom_y': 18, 'next_next_pipe_dist_to_player': 385.0, 'next_next_pipe_top_y': 80, 'next_next_pipe_bottom_y': 180}.keys()]
#conjunto_valores.append(atributos)
def jugar(n_veces=10,conjunto_valores=None):
    for _ in range(n_veces):
        env.reset_game()
        agent.reset()
        state_dict = env.getGameState()
        valores = [value for value in state_dict.values()]
        conjunto_valores.append(valores)  
        done = False
        total_reward_episode = 0
        print("\n--- Ejecutando agente ---")
        while not done:
            action = agent.act(state_dict)
            reward = env.act(action)
            print('premeio',reward)
            state_dict = env.getGameState()
            valores = [value for value in state_dict.values()]
            conjunto_valores.append(valores)            
            done = env.game_over()
            total_reward_episode += reward
            time.sleep(0.03)
        print(f"Recompensa episodio: {total_reward_episode}")

def guardar_estados(conjunto_valores):
    with open(file='./valores_atributos.csv',mode='a',newline='') as f:
        archivo_csv = csv.writer(f)
        archivo_csv.writerows(conjunto_valores)

jugar(n_veces=1,conjunto_valores=conjunto_valores)
guardar_estados(conjunto_valores)