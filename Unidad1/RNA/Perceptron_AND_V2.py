import random

# Inicialización de pesos y bias con valores pequeños aleatorios
w1 = random.uniform(-1, 1)  # Peso para x1
w2 = random.uniform(-1, 1)  # Peso para x2
bias = random.uniform(-1, 1)  # Bias (término independiente)
learning_rate = 0.1  # Tasa de aprendizaje

# Datos de entrenamiento para la compuerta AND
training_data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

# Función de activación: escalón de Heaviside
def activation_function(z):
    return 1 if z >= 0 else 0

# Entrenamiento del perceptrón
epochs = 10  # Número de iteraciones sobre el dataset

print(f"Pesos iniciales: w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}")

for epoch in range(epochs):
    print(f"\nÉpoca {epoch + 1}")
    
    for x1, x2, y_real in training_data:
        # Cálculo de la salida del perceptrón
        z = (w1 * x1) + (w2 * x2) + bias
        y_predicho = activation_function(z)

        # Calcular error
        error = y_real - y_predicho

        # Actualizar pesos y bias
        w1 += learning_rate * error * x1
        w2 += learning_rate * error * x2
        bias += learning_rate * error

        # Mostrar valores intermedios
        print(f"Entrada: ({x1}, {x2}) | Salida esperada: {y_real} | Salida predicha: {y_predicho} | "
              f"Error: {error} | Nuevos pesos: w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}")

# Evaluación del modelo después del entrenamiento
print("\n🔹 Resultados finales tras entrenamiento:")
for x1, x2, y_real in training_data:
    z = (w1 * x1) + (w2 * x2) + bias
    y_predicho = activation_function(z)
    print(f"Entrada: ({x1}, {x2}) → Salida esperada: {y_real} | Salida obtenida: {y_predicho}")
