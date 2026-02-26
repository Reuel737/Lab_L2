import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.random.rand(20, 4)
x = str(len(X))

y = (
    4*X[:,0]
    - 3*X[:,1]**2
    + 2*np.sin(X[:,2]*3)
    + 5*X[:,3]
    + np.random.normal(0, 0.1, len(X))
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

normalizer = layers.Normalization()
normalizer.adapt(X_train)

modelo_1 = keras.Sequential([
    normalizer,
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="linear")
])

modelo_1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

history_1 = modelo_1.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

modelo_2 = keras.Sequential([
    normalizer,
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear")
])

modelo_2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

history_2 = modelo_2.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(10,6))

# Modelo 1
plt.plot(history_1.history['loss'], label="Train - 1 Camada")
plt.plot(history_1.history['val_loss'], label="Val - 1 Camada")

# Modelo 2
plt.plot(history_2.history['loss'], label="Train - 3 Camadas")
plt.plot(history_2.history['val_loss'], label="Val - 3 Camadas")

plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.title("Treino vs Validação - Comparação dos Modelos")
plt.legend()
plt.grid(True)

plt.savefig("comparacao_"+x+".png")
plt.show()