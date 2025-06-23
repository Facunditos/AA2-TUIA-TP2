import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,Sequential,metrics
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)
y = np.array(y)
y_plano = np.reshape(-1,1)
y_max = np.max(y_plano)
y_min = np.min(y_plano)
print('q_value_min',y_min)
print('q_value_max',y_max)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
print('X_train.shape',X_train.shape)
print('X_test.shape',X_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)
# --- Definir la red neuronal ---
n_features = X_train.shape[1]
model = Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(units=2, activation='linear')
])

print(model.summary())

model.compile(optimizer='adam', loss='mse',metrics=['R2Score','MeanAbsoluteError'])

# --- Entrenar la red neuronal ---
# COMPLETAR: Ajustar hiperparámetros según sea necesario
# model.fit(X, y, ... demas opciones de entrenamiento ...)
history = model.fit(x=X_train,y=y_train, epochs=200, validation_data=(X_test, y_test))
# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento
plt.figure(figsize=(8,5))

plt.subplot(1,2,1)
#plt.title('Evolución del MSE en entrenamiento y validación en función a las épocas')
plt.plot(history.history['loss'], label='entrenamiento')
plt.plot(history.history['val_loss'], label = 'validación')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1,2,2)
#plt.title('Evolución del coef determinación en entrenamiento y validación en función a las épocas')
plt.plot(history.history['R2Score'], label='entrenamiento')
plt.plot(history.history['val_R2Score'], label = 'validación')
plt.xlabel('Epoch')
plt.ylabel('R2')
plt.legend()
plt.suptitle('Selección de features: evolución del MSE y del R2 en entrenamiento y validación en función a las épocas')
plt.tight_layout()
plt.savefig('./imagenes_entrenamiento_nn/Figure_7.png')
# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save('flappy_q_nn_model.h5')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
