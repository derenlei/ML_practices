import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,  # for demonstration, assume Llama3 is similarly wrapped
)

# Example hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
ALPHA = 0.1  # how strongly we regularize against the reference model
MAX_SEQ_LEN = 512
EPOCHS = 2

class MockPreferenceDataset(Dataset):
    """
    A mock dataset that yields a tuple: (input_ids, attention_mask, label_ids_plus, label_ids_minus).
    In a real scenario, you'd have an actual dataset of (prompt, accepted_response, rejected_response).
    """
    def __init__(self, tokenizer, num_samples=10):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.data = []
        
        for i in range(num_samples):
            # For demonstration, use dummy text
            prompt = f"Question {i}: Why is the sky blue?"
            accepted = f"Answer {i}: The sky is blue due to Raleigh scattering."
            rejected = f"Answer {i}: Purple monkeys living on Mars make it blue."
            
            # Tokenize
            prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            plus_enc = tokenizer(accepted, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            minus_enc = tokenizer(rejected, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            
            self.data.append((prompt_enc, plus_enc, minus_enc))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        prompt_enc, plus_enc, minus_enc = self.data[idx]
        return prompt_enc, plus_enc, minus_enc

def collate_fn(batch):
    # Each item in `batch` is (prompt_enc, plus_enc, minus_enc)
    # Suppose you gather them into batch-level tensors for plus and minus
    prompt_input_ids = [item[0]["input_ids"].squeeze(0) for item in batch]
    prompt_attention_masks = [item[0]["attention_mask"].squeeze(0) for item in batch]
    
    plus_input_ids = [item[1]["input_ids"].squeeze(0) for item in batch]
    plus_attention_masks = [item[1]["attention_mask"].squeeze(0) for item in batch]
    
    minus_input_ids = [item[2]["input_ids"].squeeze(0) for item in batch]
    minus_attention_masks = [item[2]["attention_mask"].squeeze(0) for item in batch]

    # Now pad them so each is same length
    # Use e.g. `torch.nn.utils.rnn.pad_sequence(...)`
    # and return them as a single batch.
    prompt_input_ids_padded = torch.nn.utils.rnn.pad_sequence(prompt_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    prompt_attention_masks_padded = torch.nn.utils.rnn.pad_sequence(prompt_attention_masks, batch_first=True, padding_value=0)
    
    plus_input_ids_padded = torch.nn.utils.rnn.pad_sequence(plus_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    plus_attention_masks_padded = torch.nn.utils.rnn.pad_sequence(plus_attention_masks, batch_first=True, padding_value=0)
    
    minus_input_ids_padded = torch.nn.utils.rnn.pad_sequence(minus_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    minus_attention_masks_padded = torch.nn.utils.rnn.pad_sequence(minus_attention_masks, batch_first=True, padding_value=0)

    return {
        "prompt_input_ids": prompt_input_ids_padded,
        "prompt_attention_mask": prompt_attention_masks_padded,
        "plus_input_ids": plus_input_ids_padded,
        "plus_attention_mask": plus_attention_masks_padded,
        "minus_input_ids": minus_input_ids_padded,
        "minus_attention_mask": minus_attention_masks_padded,
    }


# Hypothetical "Llama3" model paths
tokenizer_name = "openai/llama3-tokenizer"
model_name = "openai/llama3-finetuned"

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

# Fine-tuning model (the one we want to optimize via DPO)
model = LlamaForCausalLM.from_pretrained(model_name)
model.train()  # put in train mode

# Reference model (frozen)
reference_model = LlamaForCausalLM.from_pretrained(model_name)
reference_model.eval()  # keep it in eval mode
for param in reference_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

dataset = MockPreferenceDataset(tokenizer, num_samples=20)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


import torch
import torch.nn.functional as F

def compute_log_probs(model, input_ids, attention_mask):
    """
    Compute the sum of log probabilities of the tokens in `input_ids` under `model`,
    ignoring padded tokens based on `attention_mask`.

    Returns shape [batch_size], i.e. total log-prob per sample.
    """
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
    
    # outputs.logits.shape: [batch, seq_len, vocab_size]
    # For next-token prediction:
    logits = outputs.logits[:, :-1, :].contiguous()
    target = input_ids[:, 1:].contiguous()

    # Also shift the attention_mask for the target positions
    # Now mask.shape = [batch, seq_len-1]
    mask = attention_mask[:, 1:].contiguous()

    # CrossEntropy with reduction='none' returns a per-token loss
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # flatten
        target.view(-1),
        reduction='none'
    )

    # Reshape back to [batch, seq_len-1]
    ce_loss = ce_loss.view(batch_size, -1)

    # Zero out padded positions
    ce_loss = ce_loss * mask.float()

    # Sum over sequence dimension
    per_sample_nll = ce_loss.sum(dim=1)
    # per_sample_nll.shape: [batch_size]

    # Negative log-likelihood => log_probs = -NLL
    log_probs = -per_sample_nll
    return log_probs

for prompts_batch in dataloader:
    # prompts_batch is a dict:
    # {
    #   "prompt_input_ids": [batch_size, seq_len_prompt],
    #   "prompt_attention_mask": [batch_size, seq_len_prompt],
    #   "plus_input_ids": [batch_size, seq_len_plus],
    #   ...
    # }

    plus_input_ids = prompts_batch["plus_input_ids"]
    plus_attention_mask = prompts_batch["plus_attention_mask"]
    minus_input_ids = prompts_batch["minus_input_ids"]
    minus_attention_mask = prompts_batch["minus_attention_mask"]

    # Move to GPU if available:
    plus_input_ids = plus_input_ids.to(device)
    plus_attention_mask = plus_attention_mask.to(device)
    minus_input_ids = minus_input_ids.to(device)
    minus_attention_mask = minus_attention_mask.to(device)

    # Compute log-probs in one shot
    logp_plus = compute_log_probs(model, plus_input_ids, plus_attention_mask)
    logp_minus = compute_log_probs(model, minus_input_ids, minus_attention_mask)

    with torch.no_grad():
        ref_logp_plus = compute_log_probs(reference_model, plus_input_ids, plus_attention_mask)
        ref_logp_minus = compute_log_probs(reference_model, minus_input_ids, minus_attention_mask)

    s_plus = logp_plus - ALPHA * ref_logp_plus
    s_minus = logp_minus - ALPHA * ref_logp_minus

    diff = s_plus - s_minus
    loss = -torch.logsigmoid(diff).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

