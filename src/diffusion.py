import math
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(
        self, num_timesteps: int = 1000, cosine_s: float = 0.008, device: str = "cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._cosine_beta_schedule(s=cosine_s).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        ).to(self.device)
        self.alphas_cumprod_next = F.pad(self.alphas_cumprod[1:], (0, 1), value=0.0).to(
            self.device
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(
            self.device
        )
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrtrecipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        steps = self.num_timesteps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64, device=self.device)

        f_t = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod_cosine = f_t / f_t[0]

        alpha_bar_current = alphas_cumprod_cosine[1:]
        alpha_bar_previous = alphas_cumprod_cosine[:-1]

        betas = 1.0 - (alpha_bar_current / alpha_bar_previous)
        betas = torch.clip(betas, min=0.0001, max=0.9999)

        return betas.to(torch.float32)
