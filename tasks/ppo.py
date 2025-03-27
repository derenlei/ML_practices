import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    # Hypothetically, a model class with a value head might look like:
    # LlamaForCausalLMWithValueHead, 
)

# Example hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
CLIP_EPSILON = 0.2
MAX_SEQ_LEN = 128
EPOCHS = 2
GAMMA = 1.0  # discount factor if we do multi-step


class PromptDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.prompts = [
            f"Question {i}: Why is the sky blue?" for i in range(num_samples)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.prompts[idx]

def collate_fn(batch):
    # Just return the list of prompts; no batching needed for demonstration
    return batch



class LlamaPolicyValueModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(model_name)
        
        # A simple value head: a linear layer from hidden_dim -> 1
        hidden_size = self.llama.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        # We can get the hidden states from llama by using output_hidden_states=True
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs.hidden_states is a tuple of layer outputs; last element is final hidden state
        # shape: [batch, seq_len, hidden_size]
        final_hidden_state = outputs.hidden_states[-1]
        # For value, we might take the hidden state of the last token, or do some pooling.
        # Here we take the last token’s hidden state (for demonstration).
        last_token_state = final_hidden_state[:, -1, :]
        
        value = self.value_head(last_token_state).squeeze(-1)  # shape: [batch]
        
        # The policy logits are in outputs.logits -> shape: [batch, seq_len, vocab_size]
        # We'll return everything we need.
        return outputs.logits, value


tokenizer_name = "openai/llama3-tokenizer"
model_name = "openai/llama3-finetuned"

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# Our policy+value combined model
policy_value_model = LlamaPolicyValueModel(model_name)
policy_value_model.train()

optimizer = torch.optim.AdamW(policy_value_model.parameters(), lr=LEARNING_RATE)

dataset = PromptDataset(tokenizer, num_samples=20)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

def generate_response(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Greedy decode for demonstration
    for _ in range(max_new_tokens):
        logits, value = model(input_ids)
        # Take the last token logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Append to input_ids
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=1)
        # Stop if EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    response_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return response_text

def compute_logprobs(logits, labels):
    """
    Given logits [batch, seq_len, vocab_size] and labels [batch, seq_len],
    compute log-probs of the labels under the policy.
    We sum (or mean) over the sequence dimension.
    """
    log_probs = -F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none'
    )
    log_probs = log_probs.view(labels.size(0), labels.size(1))
    # Sum (or mean) across tokens
    return log_probs.sum(dim=1)


clip_range = CLIP_EPSILON

for epoch in range(EPOCHS):
    for prompts in dataloader:
        # "old_policy" reference for ratio calculation
        # In standard PPO, we keep a frozen copy of the policy parameters from the previous iteration
        old_model = LlamaPolicyValueModel(model_name)
        old_model.load_state_dict(policy_value_model.state_dict())
        old_model.eval()

        batch_loss = 0.0
        batch_size_actual = len(prompts)

        # We'll store data from rollouts so we can do multiple optimization epochs on them
        rollout_data = []

        for prompt in prompts:
            # 1) Generate a response
            response_text = generate_response(policy_value_model, tokenizer, prompt)
            
            # 2) Compute reward
            reward = mock_reward_function(prompt, response_text)
            
            # 3) Tokenize (prompt + response) for the logprob & value calculation
            combined_text = prompt + " " + response_text
            enc = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            input_ids = input_ids.to(next(policy_value_model.parameters()).device)
            attention_mask = attention_mask.to(next(policy_value_model.parameters()).device)
            
            # 4) Evaluate old policy (for ratio)
            with torch.no_grad():
                old_logits, old_value = old_model(input_ids, attention_mask=attention_mask)
                old_logprob_sum = compute_logprobs(old_logits, input_ids)
            
            # 5) Evaluate current policy
            logits, value = policy_value_model(input_ids, attention_mask=attention_mask)
            logprob_sum = compute_logprobs(logits, input_ids)
            
            # We'll treat the entire sequence as one "timestep" for simplicity
            advantage = (reward - value.item())  # extremely simplified: A = R - V

            # Store data for PPO update
            rollout_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_logprob_sum": old_logprob_sum,
                "old_value": old_value,
                "logprob_sum": logprob_sum,
                "value": value,
                "reward": reward,
                "advantage": advantage
            })

        # Now we do a PPO-style update on the batch
        # In typical PPO, we might do multiple epochs over rollout_data, with mini-batches, etc.
        # Here, we do one pass.

        policy_value_model.train()

        kl_coef = 0.01  # or some hyperparameter
        
        for item in rollout_data:
            # 1) PPO ratio
            ratio = torch.exp(item["logprob_sum"] - item["old_logprob_sum"])
            advantage = item["advantage"]
            
            # Clipped objective
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()
        
            # 2) Value loss
            value_target = torch.tensor(
                item["reward"], dtype=torch.float, device=item["value"].device
            )
            value_loss = F.mse_loss(item["value"], value_target)
        
            # 3) KL divergence penalty
            # We'll compute the KL between old_model and policy_value_model for the same input_ids
            with torch.no_grad():
                old_logits, _ = old_model(item["input_ids"], attention_mask=item["attention_mask"])
            new_logits, _ = policy_value_model(item["input_ids"], attention_mask=item["attention_mask"])
            
            # Convert to log-probs
            old_logprobs_dist = F.log_softmax(old_logits, dim=-1).detach()
            new_logprobs_dist = F.log_softmax(new_logits, dim=-1)
            
            # For demonstration: KL(old || new) or KL(new || old). We'll do KL(old || new):
            # Suppose old_logprobs_dist, new_logprobs_dist are [batch_size, seq_len, vocab_size]
            # same as
            # kl_loss = (old_probs_dist * torch.log(old_probs_dist / new_probs_dist)).sum(dim=-1).mean()

            old_probs_dist = old_logprobs_dist.exp()  # p(i)
            kl_elementwise = old_probs_dist * (old_logprobs_dist - new_logprobs_dist)
            kl = kl_elementwise.sum(dim=-1)   # sum over vocab dimension
            kl_loss = kl.mean()              # then average over batch and seq, for example

        
            # 4) Combine losses
            loss = policy_loss + value_loss + kl_coef * kl_loss
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


