# Logits are the model’s raw outputs (unnormalized probabilities).
# Targets are the correct token indices for each position in the sequence.

import torch

def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss for GPT-style pretraining.
    
    Args:
        logits (torch.Tensor): Logits of shape (batch_size * seq_length, vocab_size).
                               These are the model's raw outputs before applying softmax.
        targets (torch.Tensor): Targets of shape (batch_size * seq_length).
                                Each value is the index of the correct token.
    
    Returns:
        torch.Tensor: Scalar tensor representing the mean cross-entropy loss.
    """
    # Shape: (batch_size * seq_length, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim = 1)
    
    # Step 2: Gather the log probabilities of the correct classes
    # This selects log_probs[range(N), targets] where N = logits.size(0)
    # Extracts the log probability of the correct class (column) for each row instead of all rows.
    # [range(logits.size(0)), targets] does not work here!!
    target_log_probs = log_probs[range(logits.size(0)), targets]  # Shape: (batch_size * seq_length,)
    # Step 3: Compute the mean negative log-likelihood
    loss = -torch.mean(target_log_probs)  # Scalar
    return loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- 1. Dataset Preparation ---- #
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = tokenizer(text)  # Tokenize the text
        self.chunks = [
            self.tokens[i : i + seq_length] for i in range(0, len(self.tokens) - seq_length, seq_length)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


# ---- 2. GPT Model ---- #
class GPT(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, n_heads, n_layers, dropout=0.1):
        super(GPT, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, causal_mask):
        batch_size, seq_length = x.size()
        token_embeddings = self.embed_tokens(x)  # Shape: (batch_size, seq_length, d_model)
        position_embeddings = self.positional_encoding[:, :seq_length, :]
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self.transformer_blocks:
            x = block(x, memory=None, tgt_mask=causal_mask)

        logits = self.fc_out(x)  # Shape: (batch_size, seq_length, vocab_size)
        return logits


# ---- 3. Causal Mask ---- #
def generate_causal_mask(seq_length):
    mask = torch.tril(torch.ones((seq_length, seq_length))).to(torch.bool)
    return mask


# ---- 4. Training Script ---- #
def train_gpt():
    # Parameters
    vocab_size = 30522  # Example vocab size
    seq_length = 128
    d_model = 768
    n_heads = 12
    n_layers = 12
    num_epochs = 10
    batch_size = 16
    learning_rate = 5e-4

    # Sample text data (replace with real dataset)
    text = "This is a simple example of text for GPT-style training. Extend with real data!"
    tokenizer = lambda x: [ord(c) % vocab_size for c in x]  # Dummy tokenizer (replace with a real one)

    # Dataset and DataLoader
    dataset = TextDataset(text, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, Loss
    model = GPT(vocab_size, seq_length, d_model, n_heads, n_layers).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            causal_mask = generate_causal_mask(seq_length).cuda()

            # Forward pass
            logits = model(x, causal_mask)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Backward pass and optimization
            # Resets all gradients to zero, ensuring there’s no interference from previous steps.
            optimizer.zero_grad()
            # Computes the gradients for the current batch and stores them in .grad
            loss.backward()
            # Uses the .grad values to update the model parameters.
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# ---- Main Execution ---- #
if __name__ == "__main__":
    train_gpt()
