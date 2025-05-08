# Inteligencia Artificial, 2025A
## Taller evaluable, segundo corte


### Descripción General

Este taller está diseñado para evaluar el desarrollo de chatbots conversacionales utilizando modelos de lenguaje de gran tamaño (LLMs) con PyTorch. A través de 5 ejercicios prácticos, aprenderás los fundamentos para implementar, ajustar y desplegar asistentes virtuales basados en tecnologías de procesamiento de lenguaje natural, NLPs.

## Objetivos de Aprendizaje

- Comprender la arquitectura básica de los modelos transformer utilizados en chatbots
- Aprender a cargar y utilizar modelos pre-entrenados con Hugging Face y PyTorch
- Implementar técnicas para optimizar el rendimiento de modelos LLM en dispositivos con recursos limitados
- Desarrollar un sistema de diálogo completo con manejo de contexto conversacional
- Personalizar la "personalidad" y respuestas del chatbot mediante técnicas de fine-tuning

## Requisitos Previos

- Conocimientos básicos de Python
- Familiaridad con estructuras de datos y conceptos de programación orientada a objetos
- Entorno con PyTorch instalado (versión 2.0+)
- Conocimientos básicos de procesamiento de lenguaje natural
- GPU recomendada (pero no obligatoria)

---

### Ejercicio 1: Configuración del Entorno y Carga de Modelo Base

### Objetivo
Establecer el entorno de desarrollo necesario para trabajar con modelos LLM y cargar un modelo pre-entrenado utilizando las bibliotecas Transformers y PyTorch.

### Código Base para Completar

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# TODO: Configurar las variables de entorno para la caché de modelos
# Establecer la carpeta donde se almacenarán los modelos descargados
# ...

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    
    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    # TODO: Implementar la carga del modelo y tokenizador
    # Utiliza AutoModelForCausalLM y AutoTokenizer
    # ...
    
    # TODO: Configurar el modelo para inferencia (evaluar y usar half-precision si es posible)
    # ...
    
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    # TODO: Implementar la detección del dispositivo
    # ...
    
    # TODO: Si hay GPU disponible, mostrar información sobre la misma
    # ...
    
    return dispositivo

# Función principal de prueba
def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")
    
    # TODO: Cargar un modelo pequeño adecuado para chatbots (ej. Mistral-7B, GPT2, etc.)
    # ...
    
    # TODO: Realizar una prueba simple de generación de texto
    # ...

if __name__ == "__main__":
    main()
```

### Referencias
- Documentación de PyTorch: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
- Tutorial de optimización de modelos: [https://huggingface.co/docs/transformers/v4.29.1/en/performance](https://huggingface.co/docs/transformers/v4.29.1/en/performance)

---

### Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas

### Objetivo
Desarrollar las funciones necesarias para procesar la entrada del usuario, preparar los tokens para el modelo y generar respuestas coherentes.

### Código Base para Completar

```python
def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.
    
    Args:
        texto (str): Texto de entrada del usuario
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia
    
    Returns:
        torch.Tensor: Tensor de entrada para el modelo
    """
    # TODO: Implementar el preprocesamiento
    # - Añadir tokens especiales si son necesarios (ej. [BOS], [SEP])
    # - Convertir a tensor
    # - Pasar al dispositivo correspondiente
    # ...
    
    return entrada_procesada

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.
    
    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación
        
    Returns:
        str: Respuesta generada
    """
    # TODO: Implementar valores por defecto para parámetros de generación
    if parametros_generacion is None:
        parametros_generacion = {
            # Configurar parámetros como temperature, top_p, etc.
            # ...
        }
    
    # TODO: Implementar la generación de texto
    # Utilizar modelo.generate() con los parámetros adecuados
    # ...
    
    # TODO: Decodificar la salida y limpiar la respuesta
    # ...
    
    return respuesta

def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.
    
    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot
    
    Returns:
        str: Prompt formateado
    """
    # TODO: Implementar la función para crear un prompt de sistema
    # ...
    
    return prompt_sistema

# Ejemplo de uso
def interaccion_simple():
    modelo, tokenizador = cargar_modelo("gpt2")  # Cambia por el modelo que uses
    
    # TODO: Crear un prompt de sistema para definir la personalidad del chatbot
    # ...
    
    # TODO: Procesar una entrada de ejemplo
    # ...
    
    # TODO: Generar y mostrar la respuesta
    # ...
```

### Referencias
- Guía de generación de texto: [https://huggingface.co/docs/transformers/main/en/generation_strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- Técnicas de prompting: [https://www.promptingguide.ai/](https://www.promptingguide.ai/)

---

### Ejercicio 3: Manejo de Contexto Conversacional

### Objetivo
Implementar un sistema para mantener el contexto de la conversación, permitiendo al chatbot recordar intercambios anteriores y responder coherentemente a conversaciones prolongadas.

### Código Base para Completar

```python
class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """
    
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        """
        Inicializa el gestor de contexto.
        
        Args:
            longitud_maxima (int): Número máximo de tokens a mantener en el contexto
            formato_mensaje (callable): Función para formatear mensajes (por defecto, None)
        """
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado
        
    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
            
        Returns:
            str: Mensaje formateado
        """
        # TODO: Implementar un formato predeterminado para los mensajes
        # ...
    
    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        # TODO: Implementar la función para agregar mensajes al historial
        # ...
    
    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.
        
        Returns:
            str: Prompt completo para el modelo
        """
        # TODO: Implementar la construcción del prompt completo
        # ...
    
    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.
        
        Args:
            tokenizador: Tokenizador del modelo
        """
        # TODO: Implementar el algoritmo para truncar el historial
        # Considerar estrategias como:
        # - Eliminar mensajes más antiguos
        # - Resumir mensajes antiguos
        # - Mantener siempre el mensaje del sistema
        # ...

# Clase principal del chatbot
class Chatbot:
    """
    Implementación de chatbot con manejo de contexto.
    """
    
    def __init__(self, modelo_id, instrucciones_sistema=None):
        """
        Inicializa el chatbot.
        
        Args:
            modelo_id (str): Identificador del modelo en Hugging Face
            instrucciones_sistema (str): Instrucciones de comportamiento del sistema
        """
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        self.gestor_contexto = GestorContexto()
        
        # TODO: Inicializar el contexto con instrucciones del sistema
        # ...
    
    def responder(self, mensaje_usuario, parametros_generacion=None):
        """
        Genera una respuesta al mensaje del usuario.
        
        Args:
            mensaje_usuario (str): Mensaje del usuario
            parametros_generacion (dict): Parámetros para la generación
            
        Returns:
            str: Respuesta del chatbot
        """
        # TODO: Implementar el proceso completo:
        # 1. Agregar mensaje del usuario al contexto
        # 2. Construir el prompt completo
        # 3. Preprocesar la entrada
        # 4. Generar la respuesta
        # 5. Agregar respuesta al contexto
        # 6. Devolver la respuesta
        # ...

# Prueba del sistema
def prueba_conversacion():
    # TODO: Crear una instancia del chatbot
    # ...
    
    # TODO: Simular una conversación de varios turnos
    # ...
```

### Referencias
- Técnicas para manejo de contexto: [https://huggingface.co/blog/warm-up-to-ltm](https://huggingface.co/blog/warm-up-to-ltm)
- Estrategias de ventana deslizante: [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)

---

### Ejercicio 4: Optimización del Modelo para Recursos Limitados

### Objetivo
Implementar técnicas de optimización para mejorar la velocidad de inferencia y reducir el consumo de memoria, permitiendo que el chatbot funcione eficientemente en dispositivos con recursos limitados.

### Código Base para Completar

```python
from transformers import BitsAndBytesConfig
import torch.nn as nn

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.
    
    Args:
        bits (int): Bits para cuantización (4 u 8)
    
    Returns:
        BitsAndBytesConfig: Configuración de cuantización
    """
    # TODO: Implementar la configuración de cuantización
    # Utilizar BitsAndBytesConfig para cuantización de 4 u 8 bits
    # ...
    
    return config_cuantizacion

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo con optimizaciones aplicadas.
    
    Args:
        nombre_modelo (str): Identificador del modelo
        optimizaciones (dict): Diccionario con flags para las optimizaciones
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": True,
            "bits": 4,
            "offload_cpu": False,
            "flash_attention": True
        }
    
    # TODO: Implementar la carga con las optimizaciones específicas
    # ...
    
    return modelo, tokenizador

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.
    
    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    # TODO: Implementar la configuración de sliding window attention
    # ...

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo):
    """
    Evalúa el rendimiento del modelo en términos de velocidad y memoria.
    
    Args:
        modelo: Modelo a evaluar
        tokenizador: Tokenizador del modelo
        texto_prueba (str): Texto para pruebas de rendimiento
        dispositivo: Dispositivo donde se ejecutará
    
    Returns:
        dict: Métricas de rendimiento
    """
    # TODO: Implementar evaluación de:
    # - Tiempo de inferencia
    # - Uso de memoria
    # - Tokens por segundo
    # ...
    
    return metricas

# Función de demostración
def demo_optimizaciones():
    # TODO: Crear y evaluar diferentes configuraciones
    # 1. Modelo base sin optimizaciones
    # 2. Modelo con cuantización de 4 bits
    # 3. Modelo con sliding window attention
    # 4. Modelo con todas las optimizaciones
    # ...
    
    # TODO: Comparar y mostrar las métricas de rendimiento
    # ...
```

### Referencias
- Cuantización con bitsandbytes: [https://huggingface.co/docs/transformers/main/en/quantization](https://huggingface.co/docs/transformers/main/en/quantization)
- Flash Attention: [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
- Optimización de modelos en PyTorch: [https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

### Ejercicio 5: Personalización del Chatbot y Despliegue

### Objetivo
Implementar técnicas para personalizar el comportamiento del chatbot y prepararlo para su despliegue como una aplicación web simple.

### Código Base para Completar

```python
import gradio as gr
from peft import LoraConfig, get_peft_model, TaskType

def configurar_peft(modelo, r=8, lora_alpha=32):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.
    
    Args:
        modelo: Modelo base
        r (int): Rango de adaptadores LoRA
        lora_alpha (int): Escala alpha para LoRA
    
    Returns:
        modelo: Modelo adaptado para fine-tuning
    """
    # TODO: Implementar la configuración de PEFT
    # Crear LoraConfig y aplicarla al modelo
    # ...
    
    return modelo_peft

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.
    
    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    # TODO: Implementar el guardado de modelo y tokenizador
    # ...

def cargar_modelo_personalizado(ruta):
    """
    Carga un modelo personalizado desde una ruta específica.
    
    Args:
        ruta (str): Ruta del modelo
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    # TODO: Implementar la carga del modelo personalizado
    # ...
    
    return modelo, tokenizador

# Interfaz web simple con Gradio
def crear_interfaz_web(chatbot):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.
    
    Args:
        chatbot: Instancia del chatbot
        
    Returns:
        gr.Interface: Interfaz de Gradio
    """
    # TODO: Implementar la interfaz con Gradio
    # Definir función de callback para procesar entradas y generar respuestas
    # ...
    
    # TODO: Configurar la interfaz con componentes adecuados
    # ...
    
    return interfaz

# Función principal para el despliegue
def main_despliegue():
    # TODO: Cargar el modelo personalizado
    # ...
    
    # TODO: Crear instancia del chatbot
    # ...
    
    # TODO: Crear y lanzar la interfaz web
    # ...
    
    # TODO: (Opcional) Configurar parámetros para el despliegue
    # ...

if __name__ == "__main__":
    main_despliegue()
```

### Referencias
- PEFT y LoRA: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
- Gradio para interfaces web: [https://www.gradio.app/docs/](https://www.gradio.app/docs/)
- Despliegue de modelos: [https://huggingface.co/docs/transformers/main/en/model_sharing](https://huggingface.co/docs/transformers/main/en/model_sharing)

---

### Preguntas Teóricas

1. **¿Cuáles son las diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales? Explique qué tipo de modelo sería más adecuado para cada caso de uso y por qué.**

2. **Explique el concepto de "temperatura" en la generación de texto con LLMs. ¿Cómo afecta al comportamiento del chatbot y qué consideraciones debemos tener al ajustar este parámetro para diferentes aplicaciones?**

3. **Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?**

---

### Recursos Adicionales

- [Hugging Face Course](https://huggingface.co/course/chapter1/1): Curso completo sobre NLP con transformers
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html): Documentación oficial de PyTorch
- [The Attention Mechanism Explained](https://jalammar.github.io/illustrated-transformer/): Explicación visual del mecanismo de atención
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft): Guía sobre técnicas de fine-tuning eficiente
- [Transformers Optimization](https://huggingface.co/docs/transformers/main/en/performance): Guía de optimización de modelos transformer
- [Transformer explainer] (https://poloclub.github.io/transformer-explainer/): guía visual de modelos LLM que usan la arquitectura Transformer

### Criterios de Evaluación

- **Funcionalidad**: El chatbot debe ser capaz de mantener conversaciones coherentes y manejar múltiples turnos de diálogo.
- **Optimización**: Implementación correcta de las técnicas de optimización para mejorar el rendimiento.
- **Personalización**: Capacidad para ajustar la "personalidad" y comportamiento del chatbot.
- **Código**: Calidad, legibilidad y estructura del código implementado.
- **Comprensión teórica**: Respuestas a las preguntas teóricas y justificación de decisiones técnicas.

### Entrega:
enviar enlace a repo con código y respuestas a diegof.loaizab@unilibre.ed.co
