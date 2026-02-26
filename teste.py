import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Dados simulados (tipo ANSYS)
# -----------------------------

# Exemplo: x, y, z, tensão
X = np.random.rand(5000, 4)

# Target contínuo (ex: deformação)
y = (
    3*X[:,0]
    - 2*X[:,1]
    + 0.5*X[:,2]
    + 4*X[:,3]
    + np.random.normal(0, 0.05, 5000)
)

# Separação treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2️⃣ Normalização
# -----------------------------

normalizer = layers.Normalization()
normalizer.adapt(X_train)  # ⚠️ só no treino

# -----------------------------
# 3️⃣ Construção do Modelo
# -----------------------------

modelo = keras.Sequential([
    normalizer,
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear")  # regressão
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# 4️⃣ Callbacks
# -----------------------------

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="melhor_modelo.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    verbose=1
)

backup = keras.callbacks.BackupAndRestore(
    backup_dir="./backup_treino"
)

# -----------------------------
# 5️⃣ Treinamento
# -----------------------------

history = modelo.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop, reduce_lr, backup],
    verbose=1
)

# -----------------------------
# 6️⃣ Avaliação final
# -----------------------------

loss, mae = modelo.evaluate(X_test, y_test)
print("Loss (MSE):", loss)
print("MAE:", mae)