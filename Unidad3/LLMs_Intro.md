# Clase Introductoria: Modelos de Lenguaje de Gran Escala (LLMs) y Transformers en Python
# Inteligencia Artificial, 2025A
# Universidad Libre, Cali

---

## Objetivo General

Introducir a los estudiantes a los conceptos fundamentales de los Modelos de Lenguaje de Gran Escala (LLMs) y la arquitectura Transformer, brindando herramientas básicas para comprender y aplicar estos modelos en Python usando `transformers` de Hugging Face.

---

## Contenidos

### 1. Introducción a los Modelos de Lenguaje (LLMs)
- Definición y propósito
- Evolución de los modelos de lenguaje (n-gramas, RNNs, LSTMs, Transformers)
- Aplicaciones comunes: generación de texto, traducción, QA, clasificación

### 2. Qué es un Transformer
- Introducción a la arquitectura Transformer (Vaswani et al., 2017)
- Encoders vs Decoders
- Mecanismo de atención y self-attention
- Ventajas sobre arquitecturas anteriores

### 3. Herramientas y Librerías
- Hugging Face Transformers
- Tokenizers
- PyTorch o TensorFlow (solo como backend)
- `transformers`, `datasets`, `torch`

### 4. Preparación del Entorno
```bash
pip install transformers datasets torch
```

---

## Ejemplo 1: Cargar y Usar un LLM Preentrenado
```python
from transformers import pipeline

# Crear un pipeline de generación de texto
generator = pipeline("text-generation", model="gpt2")
result = generator("Érase una vez un profesor muy curioso que", max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

### Conceptos:
- Pipeline
- Modelo preentrenado
- Tokenización implícita

---

## Ejemplo 2: Clasificación de Sentimientos
```python
classifier = pipeline("sentiment-analysis")
print(classifier("Este curso de inteligencia artificial es increíble!"))
```

### Conceptos:
- Fine-tuning de modelos
- Transferencia de aprendizaje

---

## Ejemplo 3: Uso Directo de Tokenizer y Modelo
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar modelo y tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenización manual
inputs = tokenizer("Hugging Face hace que el NLP sea fácil", return_tensors="pt")
outputs = model(**inputs)

# Softmax para obtener probabilidades
probs = torch.nn.functional.softmax(outputs.logits, dim=1)
print(probs)
```

### Conceptos:
- Tokenización explícita
- Tensores y logits
- Softmax y probabilidades

---

## Ejercicio Práctico (30-40 minutos)
**Objetivo:** Crear un pequeño clasificador de texto que detecte si un mensaje es positivo o negativo usando un modelo de Hugging Face.

### Requisitos:
- Usar el modelo `distilbert-base-uncased-finetuned-sst-2-english`
- Escribir una función que reciba texto de entrada y devuelva la predicción con la etiqueta correspondiente (Positive/Negative)
- Crear un script interactivo que permita probar varios textos

### Plantilla base:
```python
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

while True:
    text = input("Ingresa una oración (o escribe 'salir' para terminar): ")
    if text.lower() == 'salir':
        break
    result = sentiment_pipeline(text)[0]
    print(f"Sentimiento: {result['label']} - Confianza: {result['score']:.2f}")
```

---

## Recursos y Referencias

### Artículos Académicos
- Vaswani et al., "Attention is All You Need", 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

### Documentación y Libros
- Hugging Face Docs: [https://huggingface.co/docs](https://huggingface.co/docs)
- "The Illustrated Transformer" by Jay Alammar: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
- "Natural Language Processing with Transformers" (O'Reilly)
