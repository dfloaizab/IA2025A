import numpy as np
from sklearn.linear_model import Perceptron

# Datos de entrada (X1, X2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Etiquetas de salida (AND)
y = np.array([0, 0, 0, 1])

# Crear y entrenar el Perceptrón
model = Perceptron(max_iter=1000, eta0=0.1, random_state=0,verbose=0)

#mostrar error obtenido en cada iteración:
errors = []

for __ in range(10):
    model.partial_fit(X,y,classes= np.array([0,1]))
    #obtener error de la iteración:
    errors.append( (y!=model.predict(X)).sum() )

print(f"\n Resultado final del perceptrón:")
print(f"pesos finales para las entradas:{model.coef_}")
print(f"bias final:{model.intercept_}")
print(f"Número de iteraciones efectivas:{model.n_iter_}")

#Parámetros de confiabilidad del perceptrón:
accuracy = model.score(X,y) * 100

#model.fit(X, y)

# Predicciones
print(model.predict(X))
