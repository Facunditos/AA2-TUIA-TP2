from agentes.base import Agent
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.h5'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    def act(self, state):
        """
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        valores_entrada = []
        for value in state.values():
            valores_entrada.append(value)
        valores_entrada = np.array(valores_entrada)    
        print(valores_entrada)
        return self.model.predict([valores_entrada])
    


        
