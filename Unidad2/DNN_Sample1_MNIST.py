#DNN : Dense Neural Network : red neuronal con más de una capa densa, es decir, muchas neuronas


import tensorflow as tf
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.datasets import mnist
 
import numpy as np

#pip install --upgrade keras:

#se carga el dataset de caracteres escritos:

(X_train, y_train), (x_test, y_test) = mnist.load_data()

#normalización de los datos (escalas de grises):
X_train, x_test = X_train / 255.0, x_test / 255.0

#creación de la red neuronal que clasifica las entradas como números del 0 al 9

model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28,28)), #representar imagen como un vector de características
     keras.layers.Dense(128,activation="relu"),
     keras.layers.Dense(10,activation="softmax")] #capa de salida de 10 clases ( 1 por cada caracter) 
)

#compilación del modelo de la red neuronal:
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

#entrenamiento del modelo:
model.fit(
    X_train, y_train, epochs=5, validation_data=(x_test, y_test)
)

#Evaluación del modelo:
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Parámetros de precisión del modelo: test_accuracy:{test_accuracy} ")

