import torch
from collections import Counter

def compute_rouge(prediction, reference, n=1):
    """
    Compute ROUGE-N score.
    Args:
        prediction (list): Tokenized prediction sentence.
        reference (list): Tokenized reference sentence.
        n (int): N-gram size.
    Returns:
        rouge_n (float): ROUGE-N score.
    """
    # Generate n-grams for prediction and reference
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    pred_ngrams = Counter(ngrams(prediction, n))
    ref_ngrams = Counter(ngrams(reference, n))

    # Compute overlap
    overlap = sum((pred_ngrams & ref_ngrams).values())
    total_ref_ngrams = sum(ref_ngrams.values())
    
    # Avoid division by zero
    if total_ref_ngrams == 0:
        return 0.0
    
    # Compute ROUGE-N
    rouge_n = overlap / total_ref_ngrams
    return rouge_n

# Example usage
prediction = "the cat is on the mat".split()
reference = "the cat sat on the mat".split()

rouge_1 = compute_rouge(prediction, reference, n=1)
rouge_2 = compute_rouge(prediction, reference, n=2)
print(f"ROUGE-1: {rouge_1}, ROUGE-2: {rouge_2}")
#----------------------------------------------------------------------------------------
def lcs(prediction, reference):
    """
    Compute the Longest Common Subsequence (LCS) length between prediction and reference.
    Args:
        prediction (list): Tokenized prediction sentence.
        reference (list): Tokenized reference sentence.
    Returns:
        int: Length of LCS.
    """
    m, n = len(prediction), len(reference)
    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    return dp[m, n].item()

def compute_rouge_l(prediction, reference):
    """
    Compute ROUGE-L score.
    Args:
        prediction (list): Tokenized prediction sentence.
        reference (list): Tokenized reference sentence.
    Returns:
        dict: ROUGE-L precision, recall, and F1 score.
    """
    lcs_length = lcs(prediction, reference)
    pred_len = len(prediction)
    ref_len = len(reference)

    precision = lcs_length / pred_len if pred_len > 0 else 0.0
    recall = lcs_length / ref_len if ref_len > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# Example usage
prediction = "the cat is on the mat".split()
reference = "the cat sat on the mat".split()

rouge_l = compute_rouge_l(prediction, reference)
print(f"ROUGE-L: {rouge_l}")


#----------------------------------------------------------------------------

def compute_bleu(prediction, reference, max_n=4, smooth=False):
    """
    Compute BLEU score.
    Args:
        prediction (list): Tokenized prediction sentence.
        reference (list): Tokenized reference sentence.
        max_n (int): Maximum n-gram size.
        smooth (bool): Whether to apply smoothing for zero counts.
    Returns:
        bleu (float): BLEU score.
    """
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(ngrams(prediction, n))
        ref_ngrams = Counter(ngrams(reference, n))
        
        # Compute overlap
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total_pred_ngrams = sum(pred_ngrams.values())
        
        # Avoid division by zero
        if total_pred_ngrams == 0:
            precision = 0.0
        else:
            precision = overlap / total_pred_ngrams
        
        # Smoothing for zero counts
        if smooth and precision == 0.0:
            precision = 1e-9  # Small positive value
        
        precisions.append(precision)
    
    # Brevity penalty
    pred_len = len(prediction)
    ref_len = len(reference)
    brevity_penalty = torch.exp(torch.min(torch.tensor(0.0), 1 - ref_len / pred_len))
    
    # Compute BLEU score
    bleu = brevity_penalty * torch.exp(torch.tensor(precisions).log().mean())
    return bleu.item()

# Example usage
prediction = "the cat is on the mat".split()
reference = "the cat sat on the mat".split()

bleu = compute_bleu(prediction, reference)
print(f"BLEU: {bleu}")
