#Prueba de K Nearest Neighbours como método clasificador supervisado para el dataset de life expectancy de la WHO
#Permite CLASIFICAR paises de acuerdo a su esperanza de vida de acuerdo a países similares

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#cargar dataset:
df = pd.read_csv("LifeExpectancy.csv")

#seleccionar columnas relevantes:
features = ['Life expectancy', 'Adult Mortality', 'Alcohol', 'GDP', 'Schooling']
df = df[features + ["Status"]].dropna() #Eliminamos filas con valores nulos o vacíos


#convertir "status" a columna numérica:
# 0 = "Developing", 1 = "Developed"
label_encoder = LabelEncoder()
df['Status'] = label_encoder.fit_transform(df['Status'])

#dividir en conjuntos de entrenamiento y prueba:
X_train, X_test, y_train, y_test = train_test_split(df[features],df["Status"],test_size=0.2,random_state=42)

#Normalizar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Aplicar KNN con k = un valor dado (por ejemplo 5)
#Crea el clasificador de acuerdo a las características dadas:
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train,y_train)

#Predicciones
y_pred = knn_classifier.predict(X_test)

#Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precición del clasificador:{accuracy}")

#Matriz de confusión:
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)
