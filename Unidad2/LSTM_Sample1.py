#SISTEMA SENCILLO DE NLP

import torch
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#1 paso, obtener datos de entrenamiento de ejemplo (!!!):
train_data = [
    ("It was an excelent lecture, I would repeat",1),
    ("Incomplete, not recommended",0),
    ("Looking forward to the second part of the lecture",1),
    ("Somewhat confusing, it would be better",0)
]

#2. Tokenización de las frases de entrenamiento:
tokenizer = get_tokenizer("basic_english")

#2.1. Definir el tokenizador como un "lazy generator":
#(entrega un token a la vez con "yield" y se interrumpe hasta que le pidan el próximo):
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


#3. crear vocabulario:
vocab = build_vocab_from_iterator(yield_tokens(train_data),specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

#4. convertir texto a "tensores" (vectores de características)

#Codificar texto a vectores enteros de carectarísticas:
def encode(text):
    return torch.tensor(vocab(tokenizer(text)),dtype=torch.long)

# a partir de los tokens crea una lista de correspondencias entre los vectores de características de las frases
# de entrenamiento y sus clases de salida:
def collate_batch(batch):
    texts, labels = zip(*batch)
    encoded = [ encode(text) for text in texts  ] #List Comprehension: arma una lista a partir de un iterador
    padded = torch.nn.utils.rnn.pad_sequence(encoded,batch_first=True)
    return padded, torch.tensor(labels, dtype=torch.long)

#5. Crear el modelo como una red neuronal:
#se implementa una clase que extiende un módulo de nn de torch
class LecturePerceptionLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

#6. Entrenamiento:
model = LecturePerceptionLSTM(len(vocab), embed_dim=32, hidden_dim=64, output_dim=2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Entrenamiento con 10 épocas:
for epoch in range(10):
    for text_batch, label_batch in [collate_batch(train_data)]:
        pred = model(text_batch)
        loss = loss_fn(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


#Prueba con una función de predicción:
def predict(text):
    model.eval()
    with torch.no_grad():
        encoded = encode(text).unsqueeze(0)
        pred = model(encoded)
        return torch.argmax(pred).item()
    
#TEST:
print(predict("excelent presentation")) #Salida positiva, 1
print(predict("I recommend to repeat")) #Salida positiva, 1
print(predict("Confussing"))



