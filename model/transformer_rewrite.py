import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight =  nn.Prameters(torch.ones(dim))
    def _norm(self, x):
        # setting keepdim=True ensures the output still has shape (batch_size, seq_len, 1) rather than (batch_size, seq_len). 
        # That way, the multiplication x * torch.rsqrt(...) is correctly broadcast over the last dimension.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+ self.eps)
    def forward(self, x):
        # float() convert to float32 to od RMS calculation in higher  or more stable precision to avoid numerical issues.
        # convert back to original x (fp16, bf16, fp32)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = nn.parameters(torch.ones(dim))
        self.bias = nn.parameters(torch.zeros(dim))
    def _norm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # if unbiased = True, var = 1/(n-1) * (x-x.mean(dim=-1, keepdim=True)).sum(dim=-1,keepdim=True)
        # n-1 is called sample variance and n is called population variance. 
        # Sample variance corrects the bias that can arise when estimating the entire population from just a sample.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized
    def forward(self, x):
        return output * self.weight + welf.bias


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_head)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        assert d_model % num_head == 0

        self.depth = d_model // num_head
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.linear(d_model, d_model)
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth) #-1 is the seq_length dim
        return x.permute(0, 2, 1, 3)
    def forward(self, q, k, v, mask = none):
        batch_size = q.size(0)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_qk = qk/math.sqrt(k.size(-1))

        if mask is not None:
            scaled_qk = scaled_qk.mask_fill(mask==0, float('-inf'))
        attention_weights = F.softmax(saled_qk, -1)

        # shape: (batch_size, num_heads, seq_len_q, depth)
        scaled_attention  = torch.matmul(attention_weights, v)

        concat_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = concat_attention.view(batch_size, -1, self.d_model)
        return self.dense(concat_attention)

# incorrect implementation where each query has applied to all kv.
class GroupQueryAttention(nn.Module):
    """
    A conceptual example of Group Query Attention (GQA) in PyTorch.
    
    Args:
        d_model: Total embedding dimension of the model.
        num_heads_kv: Number of heads for keys/values.
        num_groups_q: Number of "query groups" (often fewer than num_heads_kv).
        dim_feedforward: The dimension each head uses if splitting the embedding.
                         Typically d_model // num_heads for normal Multi-Head.
        dropout: Dropout probability for attention weights.
    """
    def __init__(self, d_model, num_heads_kv, num_groups_q, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads_kv = num_heads_kv
        self.num_groups_q = num_groups_q
        assert d_model % num_heads_kv == 0, \
            "For simplicity, d_model must be divisible by num_heads_kv."
        
        # Key/Value dimension per head (common multi-head style).
        self.depth_kv = d_model // num_heads_kv
        
        # Query dimension per group (assuming we distribute d_model across the query groups).
        # If we want fewer Q groups, each group might have more dimension or a different split.
        # For a simpler approach, we still do (d_model // num_groups_q).
        assert d_model % num_groups_q == 0, \
            "d_model must be divisible by num_groups_q to simplify dimension split."
        self.depth_q = d_model // num_groups_q
        
        # Projection layers
        # For keys/values, we still produce num_heads_kv heads.
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        
        # For queries, produce num_groups_q groups.
        self.wq = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_q, x_k, x_v, mask=None):
        """
        Args:
            x_q: Query tensor of shape (batch_size, seq_len_q, d_model)
            x_k: Key tensor of shape (batch_size, seq_len_k, d_model)
            x_v: Value tensor of shape (batch_size, seq_len_k, d_model)
            mask: (optional) Boolean or float mask of shape
                  (batch_size, 1, seq_len_q, seq_len_k) or broadcastable.
                  0 or False indicates masked positions.
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        B, Lq, _ = x_q.shape
        _, Lk, _ = x_k.shape
        
        # 1) Project inputs
        #    a) Queries -> num_groups_q heads
        #    b) Keys/Values -> num_heads_kv heads
        q = self.wq(x_q)   # (B, Lq, d_model)
        k = self.wk(x_k)   # (B, Lk, d_model)
        v = self.wv(x_v)   # (B, Lk, d_model)
        
        # Reshape for multi-head style but with different # heads for Q vs. K/V.
        # Q shape: (B, Lq, num_groups_q, depth_q)
        q = q.view(B, Lq, self.num_groups_q, self.depth_q).transpose(1, 2)
        # -> (B, num_groups_q, Lq, depth_q)
        
        # K shape: (B, Lk, num_heads_kv, depth_kv)
        k = k.view(B, Lk, self.num_heads_kv, self.depth_kv).transpose(1, 2)
        # -> (B, num_heads_kv, Lk, depth_kv)
        
        # V shape: (B, num_heads_kv, Lk, depth_kv)
        v = v.view(B, Lk, self.num_heads_kv, self.depth_kv).transpose(1, 2)
        
        # 2) Compute attention. We must handle that Q has num_groups_q while K/V have num_heads_kv.
        # One approach: broadcast or replicate Q heads or attempt a mapping from groups_q -> heads_kv.
        # For demonstration, let's do a simpler approach: 
        # If num_groups_q != num_heads_kv, we can do a "tile" or "repeat" style approach,
        # or we can do a smaller attention for each Q group, ignoring mismatch.
        
        # We'll do a naive approach: for each Q group, do attention with all K/V heads aggregated.
        # Then we'll gather the results. 
        # Alternatively, you'd define a more specialized approach per your design.
        
        # We'll unify K/V into a single dimension so Q can attend "the same" or we do separate calls.
        # This is fairly open-ended based on your design. 
        # Here's a simpler approach: we treat K/V as a single big multi-head set,
        # and Q as multiple queries that attend to that. We'll just iterate group by group
        # for clarity (not super efficient, but conceptually straightforward).
        
        # We expect num_groups_q <= num_heads_kv, for instance. We'll illustrate that scenario.
        # We'll store each group's attention result in a list, then cat them. 
        # This is a demonstration, so adapt as needed.
        
        outputs = []
        for g in range(self.num_groups_q):
            # Q_g shape: (B, 1, Lq, depth_q)
            # We'll just pick group g from Q -> (B, Lq, depth_q)
            q_g = q[:, g:g+1, :, :]  # (B, 1, Lq, depth_q)
            
            # Possibly we pick a sub-block of K and V if we want. Or let them remain (B, num_heads_kv, Lk, depth_kv).
            # We'll do standard scaled dot-product attention for each group over all heads in K/V
            # but we must handle dimension mismatch: q_g has #heads=1, k,v have #heads=num_heads_kv.
            
            # Option A: broadcast q_g to match num_heads_kv:
            q_g_expanded = q_g.expand(-1, self.num_heads_kv, -1, -1)  # (B, num_heads_kv, Lq, depth_q)
            
            # Then do scaled dot product:
            attn_scores = torch.matmul(q_g_expanded, k.transpose(-1, -2))  # shape: (B, num_heads_kv, Lq, Lk)
            scale = 1.0 / math.sqrt(self.depth_q)
            attn_scores = attn_scores * scale
            
            if mask is not None:
                # mask shape must be broadcastable to (B, num_heads_kv, Lq, Lk)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Multiply by v: (B, num_heads_kv, Lq, Lk) x (B, num_heads_kv, Lk, depth_kv)
            attn_output = torch.matmul(attn_weights, v)  # (B, num_heads_kv, Lq, depth_kv)
            
            # We now have #heads = num_heads_kv, but for group g, we might want to reduce or combine them somehow.
            # A common approach is to average across these heads or do a linear transform to get back to (Lq, depth_q).
            # Let's do a simple "head merge" by concatenation or averaging. 
            
            # Suppose we do an average of the heads -> (B, 1, Lq, depth_kv)
            # or we can do attn_output.view(B, -1, Lq, something). For now let's average for simplicity:
            attn_output_merged = attn_output.mean(dim=1, keepdim=True)  # -> (B, 1, Lq, depth_kv)
            
            # Now it has shape (B, 1, Lq, depth_kv). We want to align dimension with group g's dimension (depth_q).
            # But depth_kv might not match depth_q. If they're different, you'd do a learned linear transform here.
            # We'll do a small linear to map from depth_kv -> depth_q. 
            
            # Quick param-free approach if they differ? 
            # In many GQA designs, you keep them consistent, or you define a linear. For the example, assume they're same:
            if self.depth_kv != self.depth_q:
                # Example: map attn_output_merged to depth_q
                # We'll define a small transform on the fly (not recommended in real code).
                # Real code: define a projection layer in __init__, e.g., self.kv_to_qdim = nn.Linear(depth_kv, depth_q).
                pass  # or do an actual transform.
            
            # We'll rename it for clarity.
            group_result = attn_output_merged  # shape: (B, 1, Lq, depth_kv) or depth_q if mapped
            
            outputs.append(group_result)
        
        # 3) Concatenate across the group dimension
        # outputs is a list of size self.num_groups_q, each (B, 1, Lq, depth_kv or depth_q)
        # If we unify dimension, let's assume we want final shape (B, self.num_groups_q, Lq, depth_q).
        # We'll cat along dimension=1 (the group dimension).
        combined = torch.cat(outputs, dim=1)  # (B, num_groups_q, Lq, depth_kv or depth_q)
        
        # Suppose now we want to reshape back to (B, Lq, d_model). 
        # d_model = num_groups_q * depth_q, presumably. So:
        combined = combined.transpose(1, 2)  # -> (B, Lq, num_groups_q, depth_q)
        out = combined.reshape(B, Lq, self.num_groups_q * self.depth_q)  # -> (B, Lq, d_model)
        
        # 4) Final output projection
        out = self.out_proj(out)  # (B, Lq, d_model)
        
        return out



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_head)
        self.feed_forward = FeedForward(
            d_model=dim,
            d_ff=4 * dim,
        )
        self.attention_norm = RMSNorm(dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x),mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
