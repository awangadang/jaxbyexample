import flax.linen as nn
import jax.numpy as jnp
import jax

from flax.linen.attention import MultiHeadDotProductAttention


class EncoderLayer(nn.Module):
    dim_model: int
    dim_ff: int


class Encoder(nn.Module):
    num_heads: int
    embedding_dim: int  # output dimension
    hidden_dim: int  # qkv dimensions
    dropout_prob: float
    dim_feedforward: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        x = MultiHeadDotProductAttention(num_heads=self.num_heads)
