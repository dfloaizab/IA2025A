import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from LoadDatasets import *


#carga de los datos desde el dataset:
# [1] Usar regresión lineal para establecer una probabilidad de supervivencia de acuerdo a: 
#        edad, velocidad de impacto, uso de casco y uso de cinturon

# a. carga de datos
dataAccidents = loadAccidentDataset("accident.csv")
# b. convertir a dataframe:
dfAccidents = pd.DataFrame(dataAccidents)
#b1. limpiar dataset: eliminar
dfAccidents.info()
dfAccidents = dfAccidents.dropna() #Elimina filas del dataframe con columnas no nulas
dfAccidents.info()

# c. cambiar variables categóricas a variables numéricas:
dfAccidents["Gender"] = dfAccidents["Gender"].map({"Male":1, "Female":0})
dfAccidents["Helmet_Used"] = dfAccidents["Helmet_Used"].map({"Yes":1, "No":0})
dfAccidents["Seatbelt_Used"] = dfAccidents["Seatbelt_Used"].map({"Yes":1, "No":0})

# # #establecer tipo de datos de las demás variables (columnas)
dfAccidents = dfAccidents.astype({"Age":"int","Speed_of_Impact":"int","Survived":"int"})

#establecer conjunto de variables independientes (x):
Xd = dfAccidents[["Speed_of_Impact","Helmet_Used","Seatbelt_Used"]]

#establecer variable dependiente (o target, la que queremos predecir) (y):
y = dfAccidents["Survived"]

#entrenar el modelo:
# crear conjuntos de entrenamiento y prueba:
# OJO: ORDEN EN EL QUE SE RECIBEN LOS PARÁMETROS
#Primero conjuntos de entrenamiento y prueba de la variable X
#Luego conjuntos de entrenamiento y prueba de la variable y
X_train,X_test, y_train,  y_test = train_test_split(Xd, y, test_size=0.4,random_state=42)

#obtención del modelo:
model = LinearRegression()
model.fit(X_train, y_train)

#predición con el conjunto de prueba:
y_pred = model.predict(X_test)

#obtención del error de la predicción (mean squared error):
mse = mean_squared_error(y_test, y_pred)

print(f"Error de predicción del modelo:{mse}")

#PREDECIR CON UN NUEVO CASO (NUEVO CONJUNTO DE ENTRADA:)
#edad = 40, impacto = 60 km/h, usó casco y cinturón:
new_case = np.array([[90,0,0]])
new_prediction = model.predict(new_case)
print(f"Predicción del nuevo caso:{new_prediction}")








