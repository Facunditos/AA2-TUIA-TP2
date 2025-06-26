import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,Sequential,metrics
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'nacho_flappy_birds_q_table.pkl'  # Cambia el path si es necesario

with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Acción ejecutada en base al mayor q value
for state, q_values in q_table.items():
    X.append(state)
    accion = np.argmax(q_values)
    y.append(accion)
X = np.array(X)
y = np.array(y)



print(np.unique(y,return_counts=True))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
print('X_train.shape',X_train.shape)
print('X_test.shape',X_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)

# model_path = 'flappy_clasificacion_q_nn_model.h5'
# model = tf.keras.models.load_model(model_path)

# predicted_proba = model.predict(X_test).reshape((-1,))
# predicted_accion = predicted_proba>0.5
# print(np.unique(predicted_accion,return_counts=True))
# exit()
# --- Definir la red neuronal ---
n_features = X_train.shape[1]
n_clases = len(np.unique(y))
model = Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(units=1, activation='sigmoid')
])

print(model.summary())


model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['Accuracy'])

# --- Entrenar la red neuronal ---
# COMPLETAR: Ajustar hiperparámetros según sea necesario
# model.fit(X, y, ... demas opciones de entrenamiento ...)
history = model.fit(x=X_train,y=y_train,batch_size=64, epochs=200, validation_data=(X_test, y_test),callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=150, restore_best_weights=True))
# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento
plt.figure(figsize=(8,5))

plt.subplot(1,2,1)
#plt.title('Evolución del MSE en entrenamiento y validación en función a las épocas')
plt.plot(history.history['loss'], label='entrenamiento')
plt.plot(history.history['val_loss'], label = 'validación')
plt.xlabel('Epoch')
plt.ylabel('entropía binaria cruzada')
plt.legend()

plt.subplot(1,2,2)
#plt.title('Evolución del coef determinación en entrenamiento y validación en función a las épocas')
plt.plot(history.history['Accuracy'], label='entrenamiento')
plt.plot(history.history['val_Accuracy'], label = 'validación')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.suptitle('Evolución del error y del accuracy en entrenamiento y validación \nen función a las épocas')
plt.tight_layout()
plt.savefig('./imagenes_entrenamiento_nn/clasificacion_Figure_1.png')
# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save('flappy_clasificacion_q_nn_model.h5')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
