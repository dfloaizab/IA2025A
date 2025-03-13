import numpy as np
from sklearn.linear_model import Perceptron

# Datos de entrada (X1, X2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Etiquetas de salida (AND)
y = np.array([0, 0, 0, 1])

# Crear y entrenar el Perceptrón
model = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
model.fit(X, y)

# Predicciones
print(model.predict(X))
