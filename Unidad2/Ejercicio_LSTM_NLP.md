## Un chatbot con LSTM - Inteligencia Artificial 2025A

# Clasificador de Intenciones con LSTM en PyTorch

Este proyecto guía al lector en la construcción de un **clasificador de intenciones** usando redes neuronales recurrentes con **LSTM** para entender preguntas frecuentes en lenguaje natural.

---

## Objetivo

Construir un modelo de clasificación de texto capaz de identificar la intención de una frase corta, como las que se usan en sistemas de atención al cliente o asistentes virtuales.

---

## Requisitos

- Python 3.x
- Torch torchtext (2.2.2 y 0.17.2)
- scikit-learn (para dividir el dataset)
- nltk (opcional para tokenización básica)

Instalar los paquetes necesarios con:

```bash
pip install torch scikit-learn nltk
```

Dataset de ejemplo:
faq_data = [
    ("what is your return policy", "return_policy"),
    ("how do I return a product", "return_policy"),
    ("can I get a refund", "return_policy"),
    ("how do I change my password", "account_help"),
    ("I forgot my password", "account_help"),
    ("how can I reset my password", "account_help"),
    ("what are your opening hours", "store_info"),
    ("when do you open", "store_info"),
    ("where is your store located", "store_info"),
]

