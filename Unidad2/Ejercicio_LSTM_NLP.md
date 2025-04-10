## Un chatbot con LSTM - Inteligencia Artificial 2025A

# Clasificador de Intenciones con LSTM en PyTorch

Este proyecto guía en la construcción de un **clasificador de intenciones** usando redes neuronales recurrentes con **LSTM** para entender preguntas frecuentes en lenguaje natural.

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

## Dataset de ejemplo

Usamos un pequeño conjunto de datos con frases típicas de atención al cliente, etiquetadas según su intención:

```python
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
```

## Paso 1: Preprocesamiento

1. Crear vocabulario.
2. Mapear frases a secuencias de enteros.
3. Aplicar padding.
4. Codificar las etiquetas como números.

```python
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

# Tokenizador básico
def tokenize(text):
    return text.lower().split()

# Crear vocabulario
vocab = Counter()
for sentence, _ in faq_data:
    vocab.update(tokenize(sentence))

word2idx = {word: idx+1 for idx, (word, _) in enumerate(vocab.items())}
word2idx["<PAD>"] = 0

# Función para codificar frases
def encode(text):
    return [word2idx[token] for token in tokenize(text)]

# Codificar etiquetas
label2idx = {"return_policy": 0, "account_help": 1, "store_info": 2}

# Convertir datos
X = [torch.tensor(encode(sentence)) for sentence, _ in faq_data]
y = [label2idx[label] for _, label in faq_data]

# Padding
X_padded = pad_sequence(X, batch_first=True, padding_value=0)
y_tensor = torch.tensor(y)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_tensor, test_size=0.3)
```

---

## Paso 2: Definir el modelo LSTM

```python
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])  # Se usa la última capa oculta
```

---

## Paso 3: Entrenamiento del modelo

```python
model = IntentClassifier(vocab_size=len(word2idx), embed_dim=16, hidden_dim=32, output_dim=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(50):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
```

---

## Paso 4: Predicción (Inferencia)

```python
def predict(text):
    model.eval()
    encoded = torch.tensor(encode(text)).unsqueeze(0)
    padded = nn.functional.pad(encoded, (0, X_train.shape[1] - encoded.shape[1]), value=0)
    with torch.no_grad():
        output = model(padded)
        pred = torch.argmax(output, dim=1).item()
        return list(label2idx.keys())[list(label2idx.values()).index(pred)]

# Ejemplo
print(predict("how do I reset my password"))  # ➜ account_help
```

---

## Bonus: Sistema de respuestas automáticas

```python
responses = {
    "return_policy": "You can return your order within 30 days.",
    "account_help": "You can reset your password at the login page.",
    "store_info": "Our store opens at 9am and closes at 6pm."
}

question = "I forgot my password"
intent = predict(question)
print(responses[intent])
```

---

## Referencias

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Goldberg, Y. (2017). *Neural Network Methods in Natural Language Processing*. Morgan & Claypool.
- Jurafsky, D., & Martin, J. H. (2021). *Speech and Language Processing* (3rd ed. draft).
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/)


