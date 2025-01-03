import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


# Reward Model with Last Non-Padding Token
class RewardModel(nn.Module):
    def __init__(self, model_name: str):
        """
        Reward model based on a pre-trained LLaMA model.

        Args:
            model_name (str): The name of the pre-trained LLaMA model (e.g., "meta-llama/Llama-2-7b").
        """
        super(RewardModel, self).__init__()
        self.llama = LlamaModel.from_pretrained(model_name)
        self.config = LlamaConfig.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.config.hidden_size, 1)  # Outputs a single scalar value as the reward

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the reward model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Scalar rewards of shape (batch_size, 1).
        """
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Find the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1  # Indices of last non-padding tokens
        pooled_output = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]  # (batch_size, hidden_size)

        reward = self.reward_head(pooled_output)  # (batch_size, 1)
        return reward

# Pairwise Dataset for Bradley-Terry Training
class PairwiseRewardDataset(Dataset):
    def __init__(self, tokenizer, prompts, responses_a, responses_b, preferences, max_length=512):
        """
        Dataset for training the reward model with pairwise comparisons.

        Args:
            tokenizer: Tokenizer for the LLaMA model.
            prompts (list): List of input prompts.
            responses_a (list): List of first responses.
            responses_b (list): List of second responses.
            preferences (list): List of binary preferences (1 if A is preferred, 0 if B is preferred).
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.responses_a = responses_a
        self.responses_b = responses_b
        self.preferences = preferences
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response_a = self.responses_a[idx]
        response_b = self.responses_b[idx]
        preference = self.preferences[idx]

        # Tokenize prompt and responses
        encoded_a = self.tokenizer(
            prompt + " " + response_a,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_b = self.tokenizer(
            prompt + " " + response_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            encoded_a["input_ids"].squeeze(0),
            encoded_a["attention_mask"].squeeze(0),
            encoded_b["input_ids"].squeeze(0),
            encoded_b["attention_mask"].squeeze(0),
            torch.tensor(preference, dtype=torch.float32),
        )
      
# Training Loop with Bradley-Terry Loss
def train_reward_model(model, dataloader, optimizer, device, num_epochs=3):
    """
    Training loop for the reward model with pairwise comparisons.

    Args:
        model (RewardModel): The reward model.
        dataloader (DataLoader): Dataloader for the reward dataset.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device (CPU or GPU).
        num_epochs (int): Number of training epochs.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, preferences in dataloader:
            input_ids_a = input_ids_a.to(device)
            attention_mask_a = attention_mask_a.to(device)
            input_ids_b = input_ids_b.to(device)
            attention_mask_b = attention_mask_b.to(device)
            preferences = preferences.to(device)

            # Forward pass for both responses
            rewards_a = model(input_ids_a, attention_mask_a).squeeze(-1)  # (batch_size,)
            rewards_b = model(input_ids_b, attention_mask_b).squeeze(-1)  # (batch_size,)

            # Bradley-Terry pairwise loss
            probabilities = torch.sigmoid(rewards_a - rewards_b)
            loss = -torch.mean(preferences * torch.log(probabilities) + (1 - preferences) * torch.log(1 - probabilities))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
