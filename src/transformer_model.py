import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads  == 0, "d_model must be divisible by num_heads"

        # Initialize dimension parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head
        
        # Initialize linear layers for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention on formula: Attention = softmax(QK^T / sqrt(d_k))V.

        Args:
            Q (Tensor): The query tensor of shape (B, H, S, D/H).
            K (Tensor): The key tensor of shape (B, H, S, D/H).
            V (Tensor): The value tensor of shape (B, H, S, D/H).
            mask (Tensor, optional): The attention mask of shape (B, 1, 1, S). Defaults to None.

        Returns:
            Tensor: The output tensor after applying attention, has shape (B, H, S, D/H).
        """

        # Compute QK^T / sqrt(d_k)
        score_norm = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # -> (B, H, S, S)

        # Mask is provided to prevent attention to certain positions (e.g., padding tokens, future tokens)
        if mask is not None:
            score_norm = score_norm.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(score_norm, dim = -1)

        # Compute QK^T / sqrt(d_k) * V
        out = torch.matmul(attention_weights, V) # -> (B, H, S, D/H)

        return out 

    def forward(self, x_q, x_k, x_v, mask=None):
        # Compute Q, K, V
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_k)) # -> self.W_i(x_i) has shape (B, S, D)
        V = self.split_heads(self.W_v(x_v)) # -> Q, V, K have shape (B, H, S, D)

        # Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        attention_output = self.combine_heads(attention_output) # -> (B, S, D)

        # Final linear layer
        output = self.W_o(attention_output) # -> (B, S, D)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()

        # Initialize linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x -> (B, S, D)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out) # -> (B, S, D)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length: int = 512):
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
        self.register_buffer("out", out)
        
    def forward(self, x):
        return x + self.out[:x.size(1)].unsqueeze(0).to(x.device, dtype=x.dtype) # auto-cast "out" from (1, S, D) -> (B, S, D)

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()

        # Initialize sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_out = self.self_attention(x, x, x, mask) # -> (B, S, D)
        x =  self.norm1(x + self.dropout(attention_out)) # -> (B, S, D)
        ff_out = self.feed_forward(x) # -> (B, S, D)
        x = self.norm2(x + self.dropout(ff_out)) # -> (B, S, D)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()

        # Initialize sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attention_out = self.self_attention(x, x, x, tgt_mask) # -> (B, S, D)
        x = self.norm1(x + self.dropout(attention_out)) # -> (B, S, D)
        cross_attention_out = self.cross_attention(x, enc_out, enc_out, src_mask) # -> (B, S, D)
        x = self.norm2(x + self.dropout(cross_attention_out)) # -> (B, S, D)
        ff_out = self.feed_forward(x) # -> (B, S, D)
        x = self.norm3(x + self.dropout(ff_out)) # -> (B, S, D)
        return x


# Encoder / Decoder Stack Implementation
class EncoderStack(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout):
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
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout):
        super(DecoderStack, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        out = x

        for decoder in self.decoder_layers:
            out = decoder(out, enc_out, src_mask, tgt_mask)

        return out # -> (B, S, D)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_length=512):
        super(Transformer, self).__init__()

        # Initialize embeddings and positional encoding
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_length)

        # Initialize encoder and decoder stacks
        self.encoder_stack = EncoderStack(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder_stack = DecoderStack(d_model, num_heads, d_ff, num_layers, dropout)

        # Initialize output layer
        self.linear_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    # Mask helpers (tokens shape: (B, S))
    def make_src_mask(self, src, pad_token_id=0):
        """Create a padding mask for the input tensor.

        Args:
            src (Tensor): The input tensor of shape (B, S, D).
            pad_token_id (int, optional): The padding token ID. Defaults to 0.

        Returns:
            Tensor: The padding mask of shape (B, 1, 1, S).
        """
        mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)  # -> (B, 1, 1, S)
        return mask

    def make_tgt_mask(self, tgt, pad_token_id=0):
        """Create a combined mask (padding + causal) for the target tensor, used in the decoder.

        Args:
            tgt (Tensor): The target tensor of shape (B, T, D).
            pad_token_id (int, optional): The padding token ID. Defaults to 0.

        Returns:
            Tensor: The target mask of shape (B, 1, S, S).
        """
        seq_length = tgt.size(1)

        # Mask for future tokens
        causal = torch.tril(torch.ones(seq_length, seq_length, device=tgt.device, dtype=torch.bool)) # -> (S, S)
        
        # Mask for padding tokens (only mask padding tokens in keys (columns))
        pad_keys = self.make_src_mask(tgt, pad_token_id=pad_token_id) # -> (B, 1, 1, S)

        # Combine
        mask = causal.unsqueeze(0).unsqueeze(0) & pad_keys # (1 ,1, S, S) & (B, 1, 1, S) -> (B, 1, S, S)
        
        return mask # -> (B, 1, S, S)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src) 
        tgt_mask = self.make_tgt_mask(tgt, pad_token_id=0)

        # Encoder
        src_embed = self.dropout(self.pe(self.encoder_embedding(src)))
        enc_out = self.encoder_stack(src_embed, src_mask) # -> (B, S, D)

        # Decoder
        tgt_embed = self.dropout(self.pe(self.decoder_embedding(tgt)))
        dec_out = self.decoder_stack(tgt_embed, enc_out, src_mask, tgt_mask) # -> (B, S, D)

        # Output projection
        out = self.linear_layer(dec_out) # -> (B, T, V)

        return out