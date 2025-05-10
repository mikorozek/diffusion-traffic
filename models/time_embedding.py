import torch
import math
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, model_dim: int, time_embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.model_dim = model_dim
        self.time_embed_dim = time_embed_dim
        self.max_period = max_period
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generating sinusoidal embeddings for time
        Params:
            - time: shape(batch_size)
        Returns:
            - time_embeddings: shape(batch_size, time_embedding_dim)
        """
        device = time.device
        half_dim = self.model_dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device)
        args = time[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.model_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.time_embed(embedding)
