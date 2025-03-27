import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import chain

# Prepare Data
corpus = ["we are learning pytorch word2vec implementation",
          "word2vec is a great concept in NLP",
          "deep learning is fun and powerful"]

# Tokenize and preprocess the text
def preprocess(corpus):
    tokenized = [sentence.lower().split() for sentence in corpus]
    vocab = list(set(chain.from_iterable(tokenized)))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return tokenized, word_to_idx, idx_to_word

# Generate training data (context, target pairs)
def generate_training_data(tokenized, word_to_idx, window_size=2):
    data = []
    for sentence in tokenized:
        for idx, word in enumerate(sentence):
            for neighbor in range(-window_size, window_size + 1):
                if neighbor == 0:
                    continue
                target_idx = idx + neighbor
                if 0 <= target_idx < len(sentence):
                    context_word = word_to_idx[sentence[target_idx]]
                    target_word = word_to_idx[word]
                    data.append((context_word, target_word))
    return data

# Tokenize corpus and prepare vocabulary
tokenized_corpus, word_to_idx, idx_to_word = preprocess(corpus)
vocab_size = len(word_to_idx)
training_data = generate_training_data(tokenized_corpus, word_to_idx)

# Define the Skip-Gram Model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, context_word):
        embed = self.embedding(context_word)
        out = self.fc(embed)
        return out

# Hyperparameters
embedding_size = 50
num_epochs = 10
learning_rate = 0.01
batch_size = 4

# Initialize model, loss, and optimizer
model = Word2Vec(vocab_size, embedding_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for context_word, target_word in training_data:
        context_word = torch.tensor([context_word], dtype=torch.long)
        target_word = torch.tensor([target_word], dtype=torch.long)

        # Forward pass
        output = model(context_word)
        loss = criterion(output, target_word)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Extract and print the word embeddings
embeddings = model.embedding.weight.detach().numpy()
for word, idx in word_to_idx.items():
    print(f"{word}: {embeddings[idx]}")
