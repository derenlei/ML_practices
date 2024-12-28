def pad_collate(batch, pad_token_id=0):
    """
    Custom collate function to apply dynamic padding.

    Args:
        batch: List of sequences (each sequence is a tensor).
        pad_token_id: Token ID used for padding.

    Returns:
        Padded batch tensor and lengths of the original sequences.
    """
    # Get the length of the longest sequence in the batch
    max_length = max(len(seq) for seq in batch)

    # Pad all sequences to the same length
    padded_batch = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)

    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq

    # Return the padded batch and sequence lengths
    lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
    return padded_batch, lengths
