# lstm_language_model.py

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 1. Preparamos el corpus
# --------------------------
corpus = [
    "yo quiero comer",
    "yo quiero dormir",
    "yo quiero jugar",
    "tú quieres comer",
    "tú quieres dormir",
    "ella quiere leer",
    "él quiere estudiar"
]

# --------------------------
# 2. Vocabulario y codificación
# --------------------------
# Tokenizamos el corpus
tokens = set(word for sentence in corpus for word in sentence.split())
word2idx = {word: idx for idx, word in enumerate(tokens)}
idx2word = {idx: word for word, idx in word2idx.items()}

def encode_sentence(sentence):
    return torch.tensor([word2idx[word] for word in sentence.split()], dtype=torch.long)

# --------------------------
# 3. Crear pares (input, target)
# --------------------------
training_data = []
for sentence in corpus:
    words = sentence.split()
    inputs = encode_sentence(" ".join(words[:-1]))
    target = encode_sentence(words[-1])[None]  # El target es un solo valor
    training_data.append((inputs, target))

# --------------------------
# 4. Definimos el modelo LSTM
# --------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # Embedding (seq_len, embed_size)
        emb = self.embedding(x)
        # LSTM requiere (seq_len, batch, embed_size)
        lstm_out, _ = self.lstm(emb.view(len(x), 1, -1))
        # Tomamos solo la última salida
        out = self.fc(lstm_out[-1])
        return out

# --------------------------
# 5. Entrenamiento
# --------------------------
vocab_size = len(word2idx)
embed_size = 10
hidden_size = 20

model = LSTMLanguageModel(vocab_size, embed_size, hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamos por 100 épocas
for epoch in range(100):
    total_loss = 0
    for input_seq, target in training_data:
        optimizer.zero_grad()
        output = model(input_seq)
        loss = loss_fn(output.view(1, -1), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# --------------------------
# 6. Predicción
# --------------------------
def predict_next_word(prompt):
    model.eval()
    input_seq = encode_sentence(prompt)
    with torch.no_grad():
        output = model(input_seq)
        predicted_idx = torch.argmax(output).item()
        return idx2word[predicted_idx]

# --------------------------
# 7. Prueba de predicción
# --------------------------
test_prompt = "yo quiero"
predicted_word = predict_next_word(test_prompt)
print(f"\n🔮 Predicción para '{test_prompt}': {predicted_word}")
