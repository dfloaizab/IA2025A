# Resumen de Redes Neuronales con Deep Learning Básicas
# Inteligencia Artificial, 2025A
## Universidad Libre Cali
### Diego Fernando Loaiza Buitrago
---------------------------------------------------------------------------------------------------
## Conceptos Fundamentales

### Red Neuronal
Una red neuronal es un modelo computacional inspirado en el funcionamiento del cerebro humano. Está compuesta por unidades llamadas neuronas artificiales organizadas en capas que procesan información de manera paralela.

### Neurona Artificial
La unidad básica de procesamiento que recibe entradas, les asigna pesos, aplica una función de activación y produce una salida.

### Capas de una Red Neuronal
- **Capa de entrada**: Recibe los datos iniciales
- **Capas ocultas**: Realizan transformaciones intermedias
- **Capa de salida**: Genera las predicciones finales

### Función de Activación
Introduce no-linealidad al sistema. Algunas comunes son:
- ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Sigmoid: f(x) = 1/(1 + e^-x)
- Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x)

### Función de Pérdida
Mide el error entre la predicción y el valor real. Ejemplos:
- Error cuadrático medio (MSE)
- Entropía cruzada binaria
- Entropía cruzada categórica

### Descenso del Gradiente
Algoritmo de optimización que ajusta los pesos para minimizar la función de pérdida.

### Backpropagation
Método para calcular los gradientes de la función de pérdida con respecto a los parámetros de la red.

### Hiperparámetros
Configuraciones que determinan la estructura y entrenamiento:
- Tasa de aprendizaje
- Número de capas y neuronas
- Función de activación
- Tamaño del batch
- Épocas de entrenamiento

### Regularización
Técnicas para prevenir el sobreajuste:
- Dropout
- L1/L2 regularization
- Batch Normalization

## Ejemplo 1: Clasificación de Dígitos con MNIST

### Dataset
MNIST es un conjunto de 70,000 imágenes de dígitos escritos a mano (28x28 píxeles).

### Paso a Paso

```python
# Importar bibliotecas necesarias
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar y preparar los datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convertir las etiquetas a one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Visualizar algunos ejemplos
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Dígito: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. Crear el modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Convertir imagen 28x28 a vector 784
    Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y ReLU
    Dense(64, activation='relu'),   # Segunda capa oculta
    Dense(10, activation='softmax') # Capa de salida (10 dígitos)
])

# 4. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Resumen del modelo
model.summary()

# 6. Entrenar el modelo
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 7. Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# 8. Visualizar el rendimiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Hacer predicciones
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)

# 10. Mostrar algunas predicciones
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Real: {y_test[i]}\nPred: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Explicación
1. **Preparación de datos**: Cargamos MNIST, normalizamos los valores de píxeles (0-255 → 0-1) y convertimos las etiquetas a formato one-hot.
2. **Arquitectura de la red**: Creamos una red con una capa de aplanamiento, dos capas ocultas (128 y 64 neuronas con activación ReLU) y una capa de salida (10 neuronas con softmax).
3. **Compilación**: Utilizamos el optimizador Adam, entropía cruzada categórica como función de pérdida y precisión como métrica.
4. **Entrenamiento**: Entrenamos por 10 épocas con un tamaño de batch de 128.
5. **Evaluación**: Medimos el rendimiento en el conjunto de prueba.
6. **Visualización**: Graficamos la precisión y pérdida durante el entrenamiento.
7. **Predicción**: Realizamos y mostramos predicciones sobre algunos ejemplos.

## Ejemplo 2: Regresión con Boston Housing Dataset

### Dataset
Boston Housing contiene información sobre viviendas en Boston con variables como criminalidad, número de habitaciones, etc., y el objetivo es predecir el precio.

### Paso a Paso

```python
# Importar bibliotecas
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Cargar los datos
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# 2. Normalizar los datos (importante en regresión)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Explorar los datos
print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")
print(f"Rango de precios: ${np.min(y_train):.2f} - ${np.max(y_train):.2f}")
print(f"Precio promedio: ${np.mean(y_train):.2f}")

# 4. Crear el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Sin activación en la capa de salida (regresión)
])

# 5. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='mse',  # Error cuadrático medio
    metrics=['mae']  # Error absoluto medio
)

# 6. Entrenar el modelo
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 7. Evaluar el modelo
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Error absoluto medio en prueba: ${test_mae:.2f}")

# 8. Visualizar el rendimiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Error Absoluto Medio')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Hacer predicciones
predictions = model.predict(X_test_scaled)

# 10. Comparar predicciones con valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([0, 50], [0, 50], 'r--')  # Línea diagonal
plt.xlabel('Precio Real ($1000s)')
plt.ylabel('Precio Predicho ($1000s)')
plt.title('Predicciones vs Valores Reales')
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Histograma de errores
errors = predictions.flatten() - y_test
plt.figure(figsize=(10, 4))
plt.hist(errors, bins=30)
plt.xlabel('Error de Predicción ($1000s)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Explicación
1. **Preparación de datos**: Cargamos Boston Housing y normalizamos las características usando StandardScaler.
2. **Arquitectura de la red**: Creamos una red con dos capas ocultas (64 y 32 neuronas con ReLU) y una capa de salida lineal para regresión.
3. **Compilación**: Usamos el optimizador Adam, error cuadrático medio como función de pérdida y error absoluto medio como métrica.
4. **Entrenamiento**: Entrenamos por 100 épocas con un tamaño de batch de 32.
5. **Evaluación**: Medimos el error absoluto medio en el conjunto de prueba.
6. **Visualización**: Graficamos las métricas durante el entrenamiento, comparamos predicciones con valores reales y analizamos la distribución de errores.

## Ejercicios

### Ejercicio 1: Clasificación de Iris

El conjunto de datos Iris contiene medidas de 150 flores de tres especies diferentes (setosa, versicolor, virginica). Complete el siguiente código para crear un clasificador de especies:

```python
# Importar bibliotecas necesarias
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Cargar los datos
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)  # Reshape para OneHotEncoder

# 2. Dividir los datos en entrenamiento y prueba (completar)
# X_train, X_test, y_train, y_test = ...

# 3. Normalizar las características (completar)
# scaler = ...
# X_train_scaled = ...
# X_test_scaled = ...

# 4. Convertir etiquetas a one-hot encoding (completar)
# encoder = ...
# y_train_encoded = ...
# y_test_encoded = ...

# 5. Crear el modelo (completar)
# model = Sequential([
#    ...
# ])

# 6. Compilar el modelo (completar)
# model.compile(
#    ...
# )

# 7. Entrenar el modelo (completar)
# history = model.fit(
#    ...
# )

# 8. Evaluar el modelo
# test_loss, test_acc = ...
# print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# 9. Visualizar el rendimiento (completar)
# plt.figure(...)
# ...

# 10. Hacer predicciones y mostrar matriz de confusión (completar)
# predictions = ...
# ...
```

### Ejercicio 2: Predicción de Calidad de Vino

El dataset Wine Quality contiene características fisicoquímicas de vinos y una calificación de calidad. Complete el código para predecir la calidad del vino:

```python
# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Cargar los datos (asumiendo que ya has descargado el dataset)
# URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# Completar:
# data = pd.read_csv(...)
# X = data.drop('quality', axis=1).values
# y = data['quality'].values

# 2. Dividir los datos (completar)
# X_train, X_test, y_train, y_test = ...

# 3. Normalizar las características (completar)
# scaler = ...
# X_train_scaled = ...
# X_test_scaled = ...

# 4. Crear el modelo con regularización dropout (completar)
# model = Sequential([
#    ...
# ])

# 5. Compilar el modelo (completar)
# model.compile(
#    ...
# )

# 6. Configurar early stopping (completar)
# early_stopping = ...

# 7. Entrenar el modelo (completar)
# history = model.fit(
#    ...
# )

# 8. Evaluar el modelo
# test_loss, test_mae = ...
# print(f"Error absoluto medio en prueba: {test_mae:.4f}")

# 9. Visualizar predicciones vs valores reales (completar)
# predictions = ...
# plt.figure(...)
# ...

# 10. Analizar importancia de características (opcional)
# ...
```
#Objetivo de los ejercicios:

-> aplicar los conceptos de redes neuronales en problemas prácticos de clasificación y regresión usando datasets públicos populares.
