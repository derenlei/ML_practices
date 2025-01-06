import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): The dimensionality of the embedding space.
            max_len (int): The maximum length of the sequence to support.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) to store the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        i = torch.arange(0, d_model, dtype=torch.float).unsqueeze(0)  # Shape: (1, d_model)

        # Compute angle radians similar to numpy implementation
        angle_rads = position / torch.pow(10000, (2 * (i // 2)) / d_model)

        # Apply sine to even indices and cosine to odd indices
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # Even indices (0, 2, 4, ...)
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # Odd indices (1, 3, 5, ...)

        # Add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        # Register the positional encodings as a buffer, so they are not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Positional encoded tensor of the same shape as input.
        """
        seq_len = x.size(1)
        # Add positional encoding to the input tensor
        return x + self.pe[:, :seq_len, :]

# Example usage
if __name__ == "__main__":
    d_model = 512
    max_len = 100
    seq_len = 20
    batch_size = 32

    positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    x = torch.zeros(batch_size, seq_len, d_model)  # Example input
    output = positional_encoding(x)
    print("Output shape:", output.shape)



def positional_encoding(position, d_model):
    """
    Generates positional encoding for a sequence using sine and cosine functions.

    Args:
        position (int): Maximum sequence length (e.g., 50 for a sequence of length 50).
        d_model (int): Dimensionality of the embedding space (e.g., 512).

    Returns:
        tf.Tensor: Positional encoding matrix of shape (1, position, d_model).
    """

    # Step 1: Create a matrix of shape [position, d_model].
    #         Each row corresponds to a token's position (0 to position-1),
    #         and each column corresponds to an embedding dimension.
    # `pos` is a column vector of shape [position, 1].
    # `i` is a row vector of dimension indices (0, 1, 2, ..., d_model-1).
    pos = np.arange(position)[:, np.newaxis]  # Shape: [position, 1]
    i = np.arange(d_model)[np.newaxis, :]     # Shape: [1, d_model]

    # Step 2: Compute the scaled positional angles.
    #         Each element is calculated as:
    #         angle_rads[pos, i] = pos / (10000^(2 * (i//2) / d_model))
    angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    # Step 3: Apply sine to even indices and cosine to odd indices.
    # Even dimensions (0, 2, 4, ...) -> sine function
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Odd dimensions (1, 3, 5, ...) -> cosine function
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Step 4: Add a batch dimension for compatibility.
    # Shape: [1, position, d_model]
    pos_encoding = angle_rads[np.newaxis, ...]

    # Step 5: Convert to a TensorFlow tensor of type float32.
    return tf.cast(pos_encoding, dtype=tf.float32)


# Example usage
if __name__ == "__main__":
    seq_length = 50  # Maximum sequence length
    d_model = 512    # Dimensionality of the embedding space

    pos_enc = positional_encoding(seq_length, d_model)
    print("Positional Encoding shape:", pos_enc.shape)
