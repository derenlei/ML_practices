import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Example Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch, tokenizer, max_length=2048):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding = True,
        truncation = True,
        max_length = max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return encoded, labels

# Example data
texts = ["The cat is on the mat", "The quick brown fox jumps over the lazy dog"]
labels = [0, 1]  # Example binary classification labels

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = TextDataset(texts, labels)
dataloader = DataLoader(
    dataset = dataset,
    batch_size = 16,
    shuffle = True
    collate_fn = lambda batch: collate_fn(batch, tokenizer, max_length = 2048)
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_vailable() else "CPU")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr="5e-5")

num_epochs = 3
model.train()
for e in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        encoded, label = batch
        encoded = {key: val.to(device) for key, val in encoded.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(**encoded, labels=labels)
        loss = output.loss
        logits = output.logits
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss / len(dataloader)
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Load checkpoint
checkpoint = torch.load("bert_checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
