from agentes.base import Agent
from agentes.dq_agent import QAgent
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

class NNAgent(QAgent):
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
        discrete_state = self.discretize_state(state)
        valores_entrada = np.array(discrete_state)    
        valores_entrada = valores_entrada.reshape(1,len(discrete_state))
        predicted_q_values = self.model.predict(valores_entrada)[0]
        return self.actions[np.argmax(predicted_q_values)]
    


        
