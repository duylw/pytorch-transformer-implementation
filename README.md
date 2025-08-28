# Simple Transformer (PyTorch)

This project provides a simple, educational implementation of the Transformer architecture from scratch using Python and PyTorch.

Minimal from-scratch implementation of an encoder–decoder Transformer with:
- Custom Multi-Head Attention (batch-first, (B,S,D))
- Sinusoidal positional encoding
- Encoder / Decoder stacks with residual + LayerNorm (Post-LN)
- Source padding mask and target causal+padding mask
- Modular build function
- Basic unit test (shape smoke test)

## Project Structure
```
src/
  transformer_model.py
tests/
  test_transformer.py
```

## Installation
```bash
python -m venv .env
.\.env\Scripts\activate
python -m pip install -r requirements.txt
```

## Core Classes
- LayerNorm: Manual LayerNorm (feature-wise)
- MultiHeadAttention: Q,K,V projections + scaled dot-product attention
- PositionwiseFeedForward: FFN with ReLU + dropout
- PositionalEncoding: Sinusoidal (no gradients required)
- Encoder / Decoder / *Stack: N-layer stacks
- Helper: Mask builders
- Transformer: Full model wrapper

## Masks
```python
from src.transformer_model import Helper
src_mask = Helper.make_src_mask(src_tokens)       # (B,1,1,S_src)
tgt_mask = Helper.make_tgt_mask(tgt_tokens)       # (B,1,S_tgt,S_tgt)
```
Source mask hides pad tokens (key positions). Target mask combines causal (no future lookahead) + key padding.

## Usage
```python
import torch
from src.transformer_model import Transformer, Helper

model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_layers=4,
    dropout=0.1,
    max_length=128,
    tgt_max_length=128,
)

B = 2
src = torch.randint(0, 1000, (B, 20))
tgt = torch.randint(0, 1000, (B, 16))

src_mask = Helper.make_src_mask(src)
tgt_mask = Helper.make_tgt_mask(tgt)

logits = model(src, tgt, src_mask, tgt_mask)  # (B, 16, vocab)
```

## Training Loop (Currently working on it)


## Design Notes
- Masks treat True/1 as keep, 0 as masked.
- Padded query positions are not explicitly zeroed; ignored via loss masking.
- -1e9 used for masked logits.

## Reference
Attention Is All You Need