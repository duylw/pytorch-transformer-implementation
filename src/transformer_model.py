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
        batch_size, seq_length, _ = x.size()

        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()

        # After transpose, data underlying the tensor is not contiguous
        # Use contiguous() to ensure the data is contiguous in memory

        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, V, K have shape (batch_size, num_heads, seq_length, d_k)
        
        # Compute QK^T / sqrt(d_k)
        score_norm = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # score_norm has shape (batch_size, num_heads, seq_length, seq_length)

        # Mask is provided to prevent attention to certain positions (e.g., padding tokens, future tokens)
        if mask is not None:
            score_norm = score_norm.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(score_norm, dim = -1)

        # Compute QK^T / sqrt(d_k) * V
        out = torch.matmul(attention_weights, V)

        # out has shape (batch_size, num_heads, seq_length, d_k)
        return out

            
    def forward(self, x_q, x_k, x_v, mask=None):
        # Compute Q, K, V
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_k))
        V = self.split_heads(self.W_v(x_v))

        # Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        attention_output = self.combine_heads(attention_output)

        # Final linear layer
        output = self.W_o(attention_output)

        return output

class PositionalWiseFeedFoward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionalWiseFeedFoward, self).__init__()

        # Initialize linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x has shape (batch_size, seq_length, d_model)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # out has shape (batch_size, seq_length, d_model)
        return out

class SimpleEncoding(nn.Module):
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length: int = 512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length

        # X has shape (batch_size, seq_length, d_model)

        # Pos has shape (max_length, 1)
        pos = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)

        # i and angle_rates have shape (1, d_model)
        i = torch.arange(0, self.d_model, dtype=torch.float).unsqueeze(0)
        angle_rates = torch.pow(10000.0, -(2 * torch.floor(i / 2) / self.d_model))

        # Angles has shape (max_length, d_model)
        angles = pos * angle_rates

        out = torch.zeros(self.max_length, self.d_model)
        out[:, 0::2] = torch.sin(angles[:, 0::2])
        out[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer("out", out)
        
    def forward(self, x):
        seq_length = x.size(1)
        return x + self.out[:seq_length].unsqueeze(0).to(x.device, dtype=x.dtype) # auto-cast from (1, S, D) -> (B, S, D)

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionalWiseFeedFoward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_out = self.self_attention(x, x, x, mask)
        x =  self.norm1(x + self.dropout(attention_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = PositionalWiseFeedFoward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attention_out = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_out))
        cross_attention_out = self.cross_attention(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
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
        
        return out

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

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_length=512):
        super(Transformer, self).__init__()
      
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_length)

        self.encoder_stack = EncoderStack(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder_stack = DecoderStack(d_model, num_heads, d_ff, num_layers, dropout)

        self.linear_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, x):
        seq_length = x.size(1)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def create_padding_mask(self, x, pad_token_id=0):
        mask = (x != pad_token_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        return mask

    def create_combined_mask(self, tgt, pad_token_id=0):
        # Shape batch, 1, seq, seq
        mask = self.create_padding_mask(tgt, pad_token_id=pad_token_id) & self.create_causal_mask(tgt)
        return mask

    def forward(self, src, tgt):
        src_mask = self.create_padding_mask(src) 
        tgt_mask = self.create_combined_mask(tgt, pad_token_id=0)

        # Encoder
        src_embed = self.dropout(self.pe(self.encoder_embedding(src)))
        enc_out = self.encoder_stack(src_embed, src_mask)

        # Decoder
        tgt_embed = self.dropout(self.pe(self.decoder_embedding(tgt)))
        dec_out = self.decoder_stack(tgt_embed, enc_out, src_mask, tgt_mask)

        # Output projection
        out = self.linear_layer(dec_out)

        return out