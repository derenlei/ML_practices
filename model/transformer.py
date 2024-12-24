# Add RMS NORM and replace norm1 and norm2
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight =  nn.Prameters(torch.ones(dim))
    def _norm(self, x):
        # setting keepdim=True ensures the output still has shape (batch_size, seq_len, 1) rather than (batch_size, seq_len). 
        # That way, the multiplication x * torch.rsqrt(...) is correctly broadcast over the last dimension.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+ self.eps)
    def forward(self, x):
        # float() convert to float32 to od RMS calculation in higher  or more stabl eprecision to avoid numerical issues.
        # convert back to original x (fp16, bf16, fp32)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
    
class LayerNorm(nn.Module):
    


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_head)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
        
