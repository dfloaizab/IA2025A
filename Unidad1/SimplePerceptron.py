import numpy as np

# Función de activación: Escalón de Heaviside
def step_function(x):
    return 1 if x >= 0 else 0

# Clase del Perceptrón
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.random.rand(input_size)  # Pesos aleatorios
        self.bias = np.random.rand(1)  # Bias aleatorio
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return step_function(linear_output)

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                total_error += abs(error)

                # Ajuste de pesos y bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

            print(f"Época {epoch+1}/{self.epochs}, Error total: {total_error}")

# Datos de entrenamiento para la multiplicación binaria (0x0, 0x1, 1x0, 1x1)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas
y_train = np.array([0, 0, 0, 1])  # Salidas esperadas (producto binario)

# Inicializar y entrenar el perceptrón
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.train(X_train, y_train)

# Probar el perceptrón entrenado
for X, y in zip(X_train, y_train):
    prediction = perceptron.predict(X)
    print(f"Multiplicación {X[0]} x {X[1]} = {prediction} (Esperado: {y})")
