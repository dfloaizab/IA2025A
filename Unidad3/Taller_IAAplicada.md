# Inteligencia Artificial, 2025A
# Universidad Libre Seccional Cali

## Talleres Prácticos: Redes Neuronales e IA Aplicada

## Taller 3: Redes Neuronales para Reconocimiento de Imágenes

### Introducción

En este taller práctico se aprenderá a implementar y utilizar redes neuronales convolucionales (CNN) para el reconocimiento de imágenes. 
Las CNN son especialmente efectivas para tareas de visión por computadora debido a su capacidad para detectar patrones espaciales en los datos visuales.

### Punto 1: Preparación del entorno y datos

En este primer paso, se configura el entorno de trabajo y se importa un conjunto de datos de imágenes para su clasificación.

```python
# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# Cargamos el conjunto de datos CIFAR-10
# Este dataset contiene 60,000 imágenes a color de 32x32 píxeles en 10 categorías
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizamos los valores de píxeles al rango [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convertimos las etiquetas a formato one-hot
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Definimos los nombres de las clases para su visualización
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# TODO: Visualiza algunas imágenes del conjunto de datos para familiarizarte con ellas
# Pista: Usa plt.imshow() y plt.subplot() para mostrar varias imágenes
# Completa el código a continuación:

def visualizar_imagenes(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        # Tu código aquí: selecciona una posición en la figura, muestra la imagen
        # y añade el título con la clase correspondiente
        pass
    
    plt.tight_layout()
    plt.show()

# Llamar a la función para visualizar algunas imágenes
# visualizar_imagenes(...)
```

### Punto 2: Construcción de una CNN básica

Ahora se construye una red neuronal convolucional simple para clasificar las imágenes.

```python
def crear_modelo_cnn():
    # Creamos un modelo secuencial
    modelo = models.Sequential()
    
    # TODO: Añade las capas convolucionales y de agrupación (pooling)
    # Pista: Una arquitectura típica incluye:
    # - Capas Conv2D con activación ReLU
    # - Capas MaxPooling2D
    # - Capa Flatten para conectar con capas densas
    # - Capas densas finales con la última usando activación softmax
    
    # Añade aquí tus capas convolucionales
    modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    modelo.add(layers.MaxPooling2D((2, 2)))
    
    # Completa con más capas Conv2D y MaxPooling2D
    # ...
    
    # Añade capas densas (fully connected)
    modelo.add(layers.Flatten())
    
    # Completa con más capas densas
    # ...
    
    # Capa de salida con 10 neuronas (una por clase) y activación softmax
    modelo.add(layers.Dense(10, activation='softmax'))
    
    # Compilamos el modelo
    modelo.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return modelo

# Creamos el modelo
modelo_cnn = crear_modelo_cnn()

# Mostramos un resumen de la arquitectura del modelo
modelo_cnn.summary()
```

### Punto 3: Entrenamiento del modelo

En este paso se entrena el modelo con los datos de entrenamiento y se evalúa su rendimiento.

```python
def entrenar_modelo(modelo, train_images, train_labels, test_images, test_labels, epochs=10):
    # TODO: Entrena el modelo y guarda el historial de entrenamiento
    # Pista: Usa modelo.fit() y guarda su resultado
    
    historial = modelo.fit(
        # Completa los parámetros para entrenar el modelo:
        # - Datos de entrenamiento
        # - Etiquetas
        # - Epochs
        # - Validation data para monitorear el rendimiento en datos no vistos
        pass
    )
    
    # Evaluamos el modelo con los datos de prueba
    resultados = modelo.evaluate(test_images, test_labels)
    print(f"Precisión en datos de prueba: {resultados[1]:.4f}")
    
    return historial

# Entrenamos el modelo
historial = entrenar_modelo(modelo_cnn, train_images, train_labels, test_images, test_labels)

# TODO: Visualiza el historial de entrenamiento para ver cómo evoluciona la precisión y la pérdida
# Completa el código:

def visualizar_historial(historial):
    # Tu código aquí para graficar la precisión y pérdida durante el entrenamiento
    pass
```

### Punto 4: Predicciones y evaluación visual

Ahora se utiliza el modelo entrenado para hacer predicciones y visualizar los resultados.

```python
def predecir_y_visualizar(modelo, images, labels, class_names, num_imagenes=5):
    # Seleccionamos algunas imágenes aleatorias
    indices = np.random.choice(len(images), num_imagenes, replace=False)
    
    # Predecimos las clases
    predicciones = modelo.predict(images[indices])
    clases_predichas = np.argmax(predicciones, axis=1)
    clases_reales = np.argmax(labels[indices], axis=1)
    
    # TODO: Visualiza las imágenes, las predicciones y las etiquetas reales
    # Pista: Muestra cada imagen con su etiqueta real y la predicción
    # Usa colores diferentes para indicar aciertos y fallos
    
    plt.figure(figsize=(12, 4 * num_imagenes))
    for i, idx in enumerate(indices):
        # Tu código aquí: muestra la imagen, la predicción y la etiqueta real
        pass
    
    plt.tight_layout()
    plt.show()

# Llamamos a la función para visualizar algunas predicciones
# predecir_y_visualizar(...)

# Calculamos la matriz de confusión
def mostrar_matriz_confusion(modelo, images, labels, class_names):
    # TODO: Calcula y visualiza la matriz de confusión
    # Pista: Usa sklearn.metrics.confusion_matrix y matplotlib para visualizarla
    
    # Tu código aquí
    pass
```

### Punto 5: Mejoras y experimentación

Finalmente, se experimenta con técnicas para mejorar el rendimiento del modelo entrenado.

```python
def crear_modelo_mejorado():
    # TODO: Crea un modelo CNN mejorado con:
    # - Más capas o filtros
    # - Regularización (Dropout, BatchNormalization)
    # - Optimizadores o tasas de aprendizaje personalizadas
    
    modelo = models.Sequential()
    
    # Tu código aquí para crear un modelo más avanzado
    
    return modelo

# Opcional: Implementa técnicas de data augmentation para mejorar la generalización
def aplicar_data_augmentation():
    # TODO: Crea un generador de imágenes con aumentación de datos
    # Pista: Usa ImageDataGenerator de Keras
    
    # Tu código aquí
    pass

# TODO: Experimenta con diferentes hiperparámetros y compara los resultados
def experimentar_hiperparametros():
    # Tu código aquí para probar diferentes configuraciones
    pass
```

## Taller 4: Integración de Reconocimiento de Imágenes con NLP y Chatbot

### Introducción

En este taller, se desarrolla un sistema integrado que combina reconocimiento de imágenes con procesamiento de lenguaje natural (NLP) para crear un asistente virtual que puede "ver" y "conversar". 
El sistema podrá analizar imágenes y responder preguntas sobre ellas, siendo útil como herramienta de apoyo para estudiantes o consultas psicológicas básicas.

### Punto 1: Configuración del entorno y modelos base

Primero, se configura un entorno con las herramientas necesarias para trabajar con imágenes y lenguaje natural.

```python
# Importamos las librerías necesarias
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import re
import os
from PIL import Image
import io

# Descargamos recursos de NLTK necesarios para NLP
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Cargamos un modelo pre-entrenado para reconocimiento de imágenes
def cargar_modelo_imagenes():
    # Usamos MobileNetV2 pre-entrenado en ImageNet (ligero y eficiente)
    modelo_base = MobileNetV2(weights='imagenet', include_top=True)
    return modelo_base

# TODO: Implementa funciones para preprocesar imágenes
def preprocesar_imagen(ruta_imagen=None, imagen_pil=None):
    """
    Preprocesa una imagen para el modelo de reconocimiento.
    Acepta una ruta de archivo o una imagen PIL.
    """
    # Tu código aquí para cargar y preprocesar la imagen
    # Pista: Redimensiona a 224x224 y normaliza los valores
    pass

# Inicializa el lemmatizador para procesamiento de texto
lemmatizer = WordNetLemmatizer()

# Cargamos el modelo de imágenes
modelo_imagenes = cargar_modelo_imagenes()
print("Modelos base cargados correctamente.")
```

### Punto 2: Implementación del reconocimiento de imágenes

Ahora se implementa la funcionalidad para analizar y describir imágenes.

```python
def analizar_imagen(ruta_imagen=None, imagen_pil=None):
    """
    Analiza una imagen y devuelve una descripción de lo que contiene.
    """
    # Preprocesamos la imagen
    img_array = preprocesar_imagen(ruta_imagen, imagen_pil)
    
    if img_array is None:
        return "No se pudo procesar la imagen."
    
    # TODO: Implementa la predicción usando el modelo cargado
    # Pista: Usa modelo_imagenes.predict() y decode_predictions()
    
    # Tu código aquí para obtener las predicciones
    
    # Formatea los resultados para una respuesta natural
    descripcion = "En esta imagen puedo ver: "
    
    # TODO: Añade las clases detectadas a la descripción
    # Tu código aquí
    
    return descripcion

# TODO: Implementa una función para extraer características emocionales o contextuales
def interpretar_contexto_imagen(predicciones):
    """
    Interpreta el contexto emocional o situacional de una imagen
    basado en los objetos detectados.
    """
    # Tu código aquí: Analiza las predicciones para inferir el contexto
    # Por ejemplo, detectar si hay personas, animales, entornos naturales,
    # objetos que sugieran cierta actividad, etc.
    pass

# Probamos la funcionalidad
# Descomenta para probar:
# ruta_imagen_prueba = "ruta/a/una/imagen.jpg"  # Reemplaza con una ruta real
# print(analizar_imagen(ruta_imagen_prueba))
```

### Punto 3: Desarrollo del motor NLP para el chatbot

En este paso, se implementa procesamiento de lenguaje natural para el chatbot.

```python
# Definimos patrones de conversación básicos relacionados con imágenes y consultas
patrones_conversacion = {
    "saludos": [
        r"hola",
        r"buenos días",
        r"buenas tardes",
        r"hey",
        r"saludos"
    ],
    "preguntas_imagen": [
        r"qué ves en (la|esta) imagen",
        r"describe (la|esta) imagen",
        r"qué hay en (la|esta) imagen",
        r"qué puedes ver",
        r"analiza (la|esta) imagen"
    ],
    "consulta_psicologica": [
        r"me siento (triste|feliz|ansioso|preocupado|estresado)",
        r"tengo problemas con",
        r"no puedo (dormir|concentrarme|estudiar)",
        r"necesito ayuda con mis emociones",
        r"cómo puedo manejar (el estrés|la ansiedad|la depresión)"
    ],
    "consulta_academica": [
        r"no entiendo (este tema|esta materia)",
        r"cómo puedo estudiar mejor",
        r"tengo dificultades con",
        r"necesito ayuda con mis estudios",
        r"cómo mejorar (mi concentración|mi memoria|mis notas)"
    ],
    "despedida": [
        r"adiós",
        r"hasta luego",
        r"nos vemos",
        r"chao",
        r"bye"
    ]
}

# Definimos respuestas para cada categoría
respuestas = {
    "saludos": [
        "¡Hola! ¿En qué puedo ayudarte hoy?",
        "¡Saludos! Soy un asistente virtual. Puedo analizar imágenes y conversar contigo.",
        "¡Buen día! Estoy aquí para ayudarte con tus consultas e imágenes."
    ],
    "preguntas_imagen_sin_contexto": [
        "Para analizar una imagen, necesito que me la proporciones primero.",
        "No tengo ninguna imagen para analizar. ¿Podrías compartir una?",
        "Necesito ver una imagen antes de poder describirla."
    ],
    "consulta_psicologica": [
        "Entiendo cómo te sientes. ¿Podrías contarme más sobre eso?",
        "Es normal tener esos sentimientos. ¿Qué crees que los está causando?",
        "Gracias por compartir eso conmigo. Te escucho. ¿Desde cuándo te sientes así?"
    ],
    "consulta_academica": [
        "Aprender puede ser desafiante. ¿Con qué tema específico estás teniendo dificultades?",
        "Cada persona tiene su propio estilo de aprendizaje. ¿Has identificado qué métodos funcionan mejor para ti?",
        "El éxito académico requiere estrategia. ¿Has probado crear un horario de estudio?"
    ],
    "despedida": [
        "¡Hasta pronto! Fue un placer ayudarte.",
        "¡Adiós! Si necesitas más ayuda, aquí estaré.",
        "¡Que tengas un excelente día! Regresa cuando necesites apoyo."
    ],
    "default": [
        "Interesante. Cuéntame más.",
        "No estoy seguro de entender. ¿Podrías explicarlo de otra manera?",
        "Estoy aquí para ayudarte. ¿Puedes ser más específico?"
    ]
}

# TODO: Implementa la función para procesar el texto de entrada
def procesar_texto(texto):
    """
    Procesa el texto de entrada para normalización
    """
    # Tu código aquí para:
    # - Convertir a minúsculas
    # - Tokenizar
    # - Lemmatizar
    
    return []  # Devuelve tokens procesados

# TODO: Implementa la función para identificar la intención del usuario
def identificar_intencion(texto):
    """
    Identifica la intención del usuario basado en patrones
    """
    texto = texto.lower()
    
    # Tu código aquí para verificar cada patrón y determinar la categoría
    
    return "default"  # Devuelve la categoría identificada
```

### Punto 4: Integración de los componentes y creación de la interfaz

Ahora se integra el reconocimiento de imágenes con el procesamiento de lenguaje natural.

```python
class AsistenteVirtual:
    def __init__(self):
        self.modelo_imagenes = modelo_imagenes
        self.imagen_actual = None
        self.analisis_imagen_actual = None
        self.contexto_conversacion = []
        
    def procesar_imagen(self, ruta_imagen=None, imagen_pil=None):
        """
        Procesa una nueva imagen y actualiza el contexto
        """
        # TODO: Implementa la lógica para procesar una imagen y actualizar el contexto
        self.imagen_actual = None  # Almacena la imagen procesada
        self.analisis_imagen_actual = analizar_imagen(ruta_imagen, imagen_pil)
        return self.analisis_imagen_actual
    
    def responder(self, texto_usuario):
        """
        Genera una respuesta basada en el input del usuario y el contexto actual
        """
        # TODO: Implementa la lógica para generar respuestas
        # Considera:
        # - La intención del usuario
        # - El contexto de la conversación
        # - Si hay una imagen en el contexto
        
        categoria = identificar_intencion(texto_usuario)
        
        # Tu código aquí para generar una respuesta adecuada
        # basada en la categoría, el contexto y la imagen actual
        
        # Actualiza el contexto de la conversación
        self.contexto_conversacion.append({"usuario": texto_usuario, "respuesta": respuesta})
        
        return "Respuesta genérica"  # Reemplaza con tu lógica

# Creamos una interfaz simple basada en línea de comandos
def interfaz_cli():
    """
    Interfaz simple de línea de comandos para interactuar con el asistente
    """
    asistente = AsistenteVirtual()
    print("=== Asistente Virtual ===")
    print("Puedo analizar imágenes y conversar contigo.")
    print("Para analizar una imagen, escribe 'ver imagen: [ruta_a_la_imagen]'")
    print("Para salir, escribe 'salir'")
    
    while True:
        entrada = input("\nTú: ")
        
        if entrada.lower() == "salir":
            print("Asistente: ¡Hasta pronto!")
            break
        
        # TODO: Implementa la lógica para detectar si el usuario quiere analizar una imagen
        # y llama a la función correspondiente
        
        if entrada.lower().startswith("ver imagen:"):
            # Extraer la ruta de la imagen
            # Llamar a asistente.procesar_imagen()
            pass
        else:
            respuesta = asistente.responder(entrada)
            print(f"Asistente: {respuesta}")

# TODO: Implementa una interfaz gráfica básica con Tkinter o Streamlit (opcional)
def interfaz_grafica():
    """
    Interfaz gráfica simple para el asistente virtual
    """
    # Tu código aquí para crear una interfaz gráfica
    pass
```

### Punto 5: Implementación de funcionalidades avanzadas y casos de uso específicos

Finalmente, se implementan las funciones avanzadas y se adapta el asistente para casos de uso específicos.

```python
# Implementación de funcionalidades específicas para educación
def generador_ejercicios(tema, dificultad):
    """
    Genera ejercicios educativos basados en un tema y nivel de dificultad
    """
    # TODO: Implementa la generación de ejercicios basados en el tema
    # Por ejemplo, problemas matemáticos, preguntas de vocabulario, etc.
    pass

# Implementación de funciones para apoyo psicológico básico
def analisis_sentimiento(texto):
    """
    Realiza un análisis simple de sentimiento en el texto del usuario
    """
    # TODO: Implementa un analizador básico de sentimiento
    # Pista: Puedes usar diccionarios de palabras positivas/negativas
    # o NLTK vader para análisis de sentimiento
    pass

# Integración de las funcionalidades avanzadas con el asistente
def mejorar_asistente():
    """
    Mejora el asistente con las nuevas funcionalidades
    """
    # TODO: Integra las nuevas funciones en la clase AsistenteVirtual
    
    # Ejemplo de cómo podría ser la estructura:
    class AsistenteVirtualMejorado(AsistenteVirtual):
        def __init__(self):
            super().__init__()
            # Inicializar componentes adicionales
            
        def analizar_sentimiento_usuario(self, texto):
            # Implementar análisis de sentimiento
            pass
        
        def recomendar_recursos(self, tema, sentimiento=None):
            # Recomendar recursos basados en el tema y el estado emocional
            pass
        
        def crear_plan_estudio(self, tema, nivel):
            # Crear un plan de estudio personalizado
            pass
    
    return AsistenteVirtualMejorado()

# Casos de uso y ejemplos
def mostrar_casos_uso():
    """
    Muestra ejemplos de cómo utilizar el asistente en diferentes escenarios
    """
    # TODO: Implementa ejemplos prácticos de uso
    
    # Caso 1: Apoyo académico
    print("=== Caso de uso: Apoyo académico ===")
    # Código de ejemplo
    
    # Caso 2: Apoyo emocional
    print("=== Caso de uso: Apoyo emocional ===")
    # Código de ejemplo
    
    # Caso 3: Análisis de imágenes para educación
    print("=== Caso de uso: Análisis de imágenes educativas ===")
    # Código de ejemplo

# TODO: Prueba el sistema completo con un ejemplo real
def prueba_sistema_completo():
    """
    Prueba el sistema completo con un ejemplo de conversación
    """
    # Tu código aquí
    pass

# Para ejecutar la interfaz, descomenta la siguiente línea:
# interfaz_cli()
```

## Conclusiones y Siguientes Pasos

Estos son algunos puntos clave que hacen parte del aprendizaje de este taller para este curso de Inteligencia Artificial y que hacen parte de los puntos
a evaluar en la actividad final, previa aclaración de las dudas que pudieran surgir.

1. Implementación de CNNs para reconocimiento de imágenes
2. Procesamiento y análisis de texto con técnicas de NLP básicas
3. Integración de diferentes tecnologías de IA en un sistema coherente
4. Adaptación de sistemas de IA para casos de uso específicos

### Posibles Extensiones

- Implementar transferencia de aprendizaje con modelos más avanzados
- Añadir capacidades de reconocimiento de emociones en imágenes
- Integrar un modelo de lenguaje más sofisticado (como BERT o GPT)
- Mejorar la interfaz de usuario con una aplicación web completa
- Añadir capacidad para procesar audio o video

### Recursos Adicionales

- TensorFlow Documentation: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
- NLTK Documentation: [https://www.nltk.org/](https://www.nltk.org/)
- Curso de Deep Learning for Computer Vision: [https://cs231n.github.io/](https://cs231n.github.io/)
- Curso de NLP: [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)
