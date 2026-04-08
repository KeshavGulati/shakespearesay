import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK_SIZE = 256
BATCH_SIZE = 64
EMB_SIZE = 256
VOCAB_SIZE = 65 # Fixed
NUM_BLOCKS = 6
device = 'cpu'

class Decoder(nn.Module):
  
  global device
  def __init__(self, T: int, h: int):
    super().__init__()

    self.T = T
    self.h = h
    self.d_k = EMB_SIZE // h

    self.W_q = nn.Linear(EMB_SIZE, self.d_k, bias=False)
    self.W_k = nn.Linear(EMB_SIZE, self.d_k, bias=False)
    self.W_v = nn.Linear(EMB_SIZE, self.d_k, bias=False)

    self.encoder_norm1 = nn.LayerNorm(self.d_k, bias=False)
    self.encoder_norm2 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm1 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm2 = nn.LayerNorm(self.d_k, bias=False)
    self.decoder_norm3 = nn.LayerNorm(self.d_k, bias=False)

    self.feed_forward = nn.Sequential(
                          nn.Linear(EMB_SIZE, 4*EMB_SIZE),
                          nn.GELU(),
                          nn.Linear(4*EMB_SIZE, EMB_SIZE)
                        )
    
    self.norm1 = nn.LayerNorm(EMB_SIZE, device=device)
    self.norm2 = nn.LayerNorm(EMB_SIZE, device=device)
    self.proj = nn.Linear(h * self.d_k, EMB_SIZE)
    
  def _self_attention(self, 
                      Q: torch.Tensor, 
                      K: torch.Tensor, 
                      V: torch.Tensor, 
                      mask: bool = False):
    
    _, T, _ = Q.shape
    a1 = torch.bmm(Q, K.transpose(1, 2)) / ((self.d_k) ** (1/2))

    if (mask):
      # masking future values
      helper = torch.tril(torch.ones((T, T), device=Q.device))
      a1 = a1.masked_fill(helper == 0, float('-inf'))

    a2 = F.softmax(a1, dim=-1)
    attention = torch.bmm(a2, V)
    return attention
  
  def _multi_head_attention(self, 
                            Q: torch.Tensor, 
                            K: torch.Tensor, 
                            V: torch.Tensor, 
                            mask: bool = False):
    
    b1 = [self._self_attention(Q, K, V, mask) for h in range(self.h)]
    b2 = torch.cat(b1, dim=-1)
    mh_attention = self.proj(b2)
    return mh_attention
  
  def _add_and_norm(self, 
                    x1: torch.Tensor, 
                    x2: torch.Tensor,
                    norm_layer: torch.nn.Module):
    
    a = x1 + x2
    return norm_layer(a)
  
  
  # def encoder(self, 
  #             x: torch.Tensor):
    
  #   Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

  #   mh_out = self._multi_head_attention(Q, K, V)
  #   add_norm_out1 = self._add_and_norm(x, mh_out)

  #   ff_out = self.feed_forward(add_norm_out1)
  #   add_norm_out2 = self._add_and_norm(add_norm_out1, ff_out)

  #   return add_norm_out2
  
  def forward(self, 
              x: torch.Tensor, 
              encoder_out: torch.Tensor | None = None
              ):
    
    Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

    mh_out1 = self._multi_head_attention(Q, K, V, mask=True)
    add_norm_out1 = self._add_and_norm(x, mh_out1, self.norm1) # (B, T, C)

    # mh_out2 = self._multi_head_attention(encoder_out, encoder_out, add_norm_out1)
    # add_norm_out2 = self._add_and_norm(add_norm_out1, mh_out2)

    ff_out = self.feed_forward(add_norm_out1)
    add_norm_out3 = self._add_and_norm(add_norm_out1, ff_out, self.norm2)

    return add_norm_out3
  
  
class ShakespeareGPT(nn.Module):
  def __init__(self, T, h, num_blocks):
    super().__init__()

    self.T = T
    self.h = h

    self.get_token_embeddings = nn.Embedding(VOCAB_SIZE, EMB_SIZE)
    self.get_pos_embeddings = nn.Embedding(self.T, EMB_SIZE)

    self.lm_head = nn.Linear(EMB_SIZE, VOCAB_SIZE)

    self.decoder_blocks = nn.ModuleList([
      Decoder(T, h) for _ in range(num_blocks)
    ])

  def forward(self, 
            X: torch.Tensor, 
            Y: torch.Tensor | None = None
            ):
  
    loss = None
    
    B, T = X.shape
    
    tok_emb = self.get_token_embeddings(X) # x.shape = B, T, C
    pos_emb = self.get_pos_embeddings(torch.arange(T, device=X.device))
    x = tok_emb + pos_emb

    # encoder_out = self.encoder(x)
    for decoder in self.decoder_blocks:
      x = decoder(x)

    logits = self.lm_head(x)
    # logits, loss = F.softmax(logits, dim=-1), None

    if Y is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      Y = Y.view(B*T)
      loss = F.cross_entropy(logits, Y)

    return logits, loss

  def generate(self, 
              idx: list[int], 
              max_new_tokens: int
              ):
    
    for _ in range(max_new_tokens):
      block_idx = idx[:, -self.T: ]
      logits, loss = self(block_idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      
      # Get index with highest probability
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx  

