import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()

        # Eps prevents the division by zero
        self.eps = eps

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True) # -> (B, S, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # -> (B, S, 1)

        # Normalize
        x_norm = (x-mean) / torch.sqrt(var + self.eps) # -> (B, S, D)

        return x_norm * self.weight + self.bias # -> (B, S, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()

        # Make sure model dimension is divisible by number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimension parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head

        # Initialize linear layers for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """Split a tensor into multiple heads.

        Args:
            x (Tensor): The input tensor of shape (B, S, D).

        Returns:
            Tensor: The input tensor split into multiple heads, has shape (B, H, S, D/H).
        """
        batch_size, seq_length, _ = x.size()

        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # -> (B, H, S, D/H)

    def combine_heads(self, x):
        """Combine multiple heads into a single tensor.

        Args:
            x (Tensor): The input tensor of shape (B, H, S, D/H).

        Returns:
            Tensor: The input tensor combined from multiple heads, has shape (B, S, D).
        """
        batch_size, _, seq_length, _ = x.size()

        # After transpose, data underlying the tensor is not contiguous
        # Use contiguous() to ensure the data is contiguous in memory

        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) # -> (B, S, D)

    # Make this function static to easily access and visualize attention scores
    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask, dropout: nn.Dropout):
        """Compute scaled dot-product attention on formula: Attention = softmax(QK^T / sqrt(d_k))V.

        Args:
            Q (Tensor): The query tensor of shape (B, H, S, D/H).
            K (Tensor): The key tensor of shape (B, H, S, D/H).
            V (Tensor): The value tensor of shape (B, H, S, D/H).
            mask (Tensor, optional): The attention mask of shape (B, 1, 1, S) broadcastable to (B,1,S,S).

        Returns:
            Tensor: The output tensor after applying attention, has shape (B, H, S, D/H).
        """

        # Compute QK^T / sqrt(d_k)
        score_norm = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1)) # -> (B, H, S, S)

        # Mask is provided to prevent attention to certain positions (e.g., padding tokens, future tokens)
        if mask is not None:
            # Mask out the invalid positions by a large negative value
            score_norm = score_norm.masked_fill(mask==0, -1e9)

        attention_weights = torch.softmax(score_norm, dim = -1) # -> (B, H, S, S)
        attention_weights = dropout(attention_weights) # -> (B, H, S, S)

        # Compute QK^T / sqrt(d_k) * V
        out = torch.matmul(attention_weights, V) # -> (B, H, S, D/H)
        
        return out, score_norm

    def forward(self, x_q, x_k, x_v, mask=None):
        # Compute Q, K, V
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_k)) # -> self.W_i(x_i) has shape (B, S, D)
        V = self.split_heads(self.W_v(x_v)) # -> Q, V, K have shape (B, H, S, D/H)

        # Compute attention
        attention_output, _ = MultiHeadAttention.scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # Combine heads
        attention_output = self.combine_heads(attention_output) # -> (B, S, D)

        # Final linear layer
        output = self.W_o(attention_output) # -> (B, S, D)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(PositionwiseFeedForward, self).__init__()

        # Initialize linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x -> (B, S, D)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # -> (B, S, D)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super(PositionalEncoding, self).__init__()

        # Initialize parameters
        self.d_model = d_model
        self.max_length = max_length

        # Create an array of position in sequence (the i-th word in sequence)
        pos = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1) # -> (S, 1)

        # Create an array of position in embed dimension (the i-th dimension in the model)
        i = torch.arange(0, self.d_model, dtype=torch.float).unsqueeze(0) # -> (1, D)
        angle_rates = torch.pow(10000.0, -(2 * torch.floor(i / 2) / self.d_model)) # -> (1, D)

        # Angles has shape (S, D)
        angles = pos * angle_rates # -> (S, D)

        out = torch.zeros(self.max_length, self.d_model)
        out[:, 0::2] = torch.sin(angles[:, 0::2])
        out[:, 1::2] = torch.cos(angles[:, 1::2]) # -> (S, D)

        # Each PositionalEncoding has Its own buffer
        self.register_buffer("out", out.unsqueeze(0)) # -> (1, S, D)
        
    def forward(self, x):
        actual_length = x.size(1)
    
        if actual_length > self.max_length:
            raise ValueError(f"Input sequence length {actual_length} exceeds maximum length {self.max_length}")

        return x + self.out[:, :actual_length, :].to(x.device, x.dtype) # -> (B, L, D)

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(Encoder, self).__init__()

        # Initialize sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_out = self.self_attention(x, x, x, mask) # -> (B, S, D)

        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(attention_out)) # -> (B, S, D)

        ff_out = self.feed_forward(x) # -> (B, S, D)

        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_out)) # -> (B, S, D)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(Decoder, self).__init__()

        # Initialize sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attention_out = self.self_attention(x, x, x, tgt_mask) # -> (B, S, D)
        
        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(attention_out)) # -> (B, S, D)

        cross_attention_out = self.cross_attention(x, enc_out, enc_out, src_mask) # -> (B, S, D)

        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(cross_attention_out)) # -> (B, S, D)

        ff_out = self.feed_forward(x) # -> (B, S, D)

        # Residual connection and layer normalization
        x = self.norm3(x + self.dropout(ff_out)) # -> (B, S, D)
        return x


# Encoder / Decoder Stack Implementation
class EncoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super(EncoderStack, self).__init__()

        self.encoder_layers = nn.ModuleList([
            Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        out = x

        for encoder in self.encoder_layers:
            out = encoder(out, mask)
        
        return out # -> (B, S, D)

class DecoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super(DecoderStack, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        out = x

        for decoder in self.decoder_layers:
            out = decoder(out, enc_out, src_mask, tgt_mask)

        return out # -> (B, S, D)

class Helper():
    def __init__(self):
        pass

    @staticmethod
    def make_src_mask(src, pad_token_id: int = 0):
        """Create a padding mask for the input tensor.

        Args:
            src (Tensor): The input tensor (tokens id) of shape (B, S).
            pad_token_id (int, optional): The padding token ID. Defaults to 0.

        Returns:
            Tensor: The padding mask of shape (B, 1, 1, S).
        """

        return (src != pad_token_id).unsqueeze(1).unsqueeze(2)  # -> (B, 1, 1, S)

    @staticmethod
    def make_tgt_mask(tgt, pad_token_id: int = 0):
        """Create a combined mask (padding + causal) for the target tensor, used in the decoder.

        Args:
            tgt (Tensor): The target tensor (tokens id) of shape (B, S).
            pad_token_id (int, optional): The padding token ID. Defaults to 0.

        Returns:
            Tensor: The target mask of shape (B, 1, S, S).
        """

        seq_length = tgt.size(1)

        # Mask for future tokens
        causal = torch.tril(torch.ones(seq_length, seq_length, device=tgt.device, dtype=torch.bool)) # -> (S, S)
        
        # Mask for padding tokens (only mask padding tokens in keys (columns))
        pad_keys = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2) # -> (B, 1, 1, S)

        # Combine
        mask = causal.unsqueeze(0).unsqueeze(0) & pad_keys # (1 ,1, S, S) & (B, 1, 1, S) -> (B, 1, S, S)
        
        return mask

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float, max_length: int, tgt_max_length: int):
        super(Transformer, self).__init__()

        # Initialize parameters
        self.d_model = d_model

        # Initialize embeddings and positional encoding
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pe = PositionalEncoding(d_model, max_length)
        self.tgt_pe = PositionalEncoding(d_model, tgt_max_length)

        # Initialize encoder and decoder stacks
        self.encoder_stack = EncoderStack(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder_stack = DecoderStack(d_model, num_heads, d_ff, num_layers, dropout)

        # Initialize output layer
        self.linear_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):

        # Encoder
        src_embed = self.dropout(self.src_pe(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_out = self.encoder_stack(src_embed, src_mask) # -> (B, S, D)

        # Decoder
        tgt_embed = self.dropout(self.tgt_pe(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_out = self.decoder_stack(tgt_embed, enc_out, src_mask, tgt_mask) # -> (B, S, D)

        # Output projection
        out = self.linear_layer(dec_out) # -> (B, S, V)

        return out