# Taller Evaluable, Corte 1,  Inteligencia Artificial, 2025A

## Temas Cubiertos:
1. Regresión Lineal
2. K-Nearest Neighbors (KNN)
3. K-Means Clustering
4. Análisis de Componentes Principales (PCA)
5. Perceptrón Simple
6. Redes Neuronales con 1-2 capas ocultas

## Instrucciones:
- Complete el código en los espacios indicados.
- Responda las preguntas conceptuales.
- Consulte las referencias sugeridas para reforzar tu aprendizaje.

---

### 1. Regresión Lineal
#### a) Complete el código para entrenar un modelo de regresión lineal en Python con scikit-learn:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 7, 11])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(____, ____)

# Predicción
X_pred = np.array([6]).reshape(-1, 1)
prediction = model.predict(____)
print(f"Predicción para X=6: {prediction}")
```

#### b) Pregunta Conceptual:
¿Qué significan los coeficientes del modelo de regresión lineal?

---

### 2. K-Nearest Neighbors (KNN)
#### a) Complete el código para clasificar puntos usando KNN:
```python
from sklearn.neighbors import KNeighborsClassifier

# Datos de ejemplo
X_train = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [6, 5]])
y_train = np.array([0, 0, 0, 1, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(____, ____)

# Predicción para un nuevo punto
new_point = np.array([[4, 3]])
pred = knn.predict(____)
print(f"Clase predicha: {pred}")
```

#### b) Pregunta Conceptual:
¿Cómo afecta el valor de `k` al modelo?

---

### 3. K-Means Clustering
#### a) Complete el código para realizar clustering con K-Means:
```python
from sklearn.cluster import KMeans

X = np.array([[1, 2], [2, 3], [3, 1], [8, 8], [9, 9], [10, 10]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(____)
print("Centroides:", kmeans.cluster_centers_)
print("Etiquetas asignadas:", kmeans.labels_)
```

#### b) Pregunta Conceptual:
¿Qué significa el número de clusters en K-Means?

---

### 4. Análisis de Componentes Principales (PCA)
#### a) Complete el código para reducir la dimensionalidad con PCA:
```python
from sklearn.decomposition import PCA

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(____)
print("Datos reducidos:", X_reduced)
```

#### b) Pregunta Conceptual:
¿Cómo se interpretan los componentes principales?

---

### 5. Perceptrón Simple
#### a) Complete el código para entrenar un perceptrón en Python:
```python
from sklearn.linear_model import Perceptron

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])  # AND lógico

perceptron = Perceptron()
perceptron.fit(____, ____)

# Predicción
print(perceptron.predict([[0, 1], [1, 1]]))
```

#### b) Pregunta Conceptual:
¿Por qué el perceptrón no puede resolver el problema XOR?

---

### 6. Redes Neuronales con 1-2 capas ocultas
#### a) Complete el código para una red neuronal con una capa oculta en Keras:
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(____, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### b) Pregunta Conceptual:
¿Qué efecto tiene aumentar el número de neuronas en una capa oculta?

---

## Referencias
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow & Keras](https://www.tensorflow.org/)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---
