# used once when init transformer model
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # tensor of frequencies, decreasing exponentially with respect to the positional encoding dimension.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # A range of values from 0 to end - 1, representing discrete time steps.
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # 
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# self.freqs_cis = precompute_freqs_cis(
#     params.dim // params.n_heads,
#     params.max_seq_len * 2,
#     params.rope_theta, # 500000
# )

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # [seq_len,1,head_dim] -> [1,seq_len,1,head_dim]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # *xq.shape[:-1] = shape1, shape2, ... to keep prev shape same in reshaping
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # [seq_len,1,head_dim] -> [1,seq_len,1,head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # (batch,seq_len,num_heads,head_dim//2,2) -> (batch,seq_len,num_heads,head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
