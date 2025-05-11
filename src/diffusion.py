import math
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn


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

        log_variance_input = torch.zeros_like(self.posterior_variance)
        log_variance_input[0] = self.posterior_variance[1]
        log_variance_input[1:] = self.posterior_variance[1:]

        self.posterior_log_variance_clipped = torch.log(
            torch.maximum(log_variance_input, torch.tensor(1e-20, device=self.device))
        ).to(self.device)

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
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

    def _extract(
        self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple
    ) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(0, t.to(a.device).long())

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        x_0_on_device = x_0.to(self.device, dtype=self.betas.dtype)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t.long(), x_0_on_device.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alpha_cumprod, t.long(), x_0_on_device.shape
        )

        x_t = (
            sqrt_alphas_cumprod_t * x_0_on_device
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

        return x_t

    def _predict_x0_from_predicted_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        x_t = x_t.to(self.device, dtype=self.betas.dtype)
        predicted_noise = predicted_noise.to(self.device, dtype=self.betas.dtype)
        t_long = t.to(self.device).long()
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t_long, x_t.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alpha_cumprod, t_long, x_t.shape
        )

        x_0_pred = sqrt_recip_alphas_cumprod_t * (
            x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise
        )

        return x_0_pred

    def _q_posterior_mean_variance(
        self, x_0_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_0_pred = x_0_pred.to(self.device, dtype=self.betas.dtype)

        x_t = x_t.to(self.device, dtype=self.betas.dtype)
        t_long = t.to(self.device).long()

        posterior_mean_coef1_t = self._extract(
            self.posterior_mean_coef1, t_long, x_t.shape
        )
        posterior_mean_coef2_t = self._extract(
            self.posterior_mean_coef2, t_long, x_t.shape
        )

        posterior_mean = (
            posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t
        )

        posterior_log_variance_t = self._extract(
            self.posterior_log_variance_clipped, t_long, x_t.shape
        )

        return posterior_mean, posterior_log_variance_t

    def p_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x_t = x_t.to(self.device, dtype=self.betas.dtype)
        t_long = t.to(self.device).long()

        predicted_noise = model(x_t, t_long)
        predicted_noise = predicted_noise.to(self.device, dtype=self.betas.dtype)

        x_0_pred = self._predict_x0_from_predicted_noise(x_t, t_long, predicted_noise)

        model_mean, model_log_variance = self._q_posterior_mean_variance(
            x_0_pred, x_t, t_long
        )

        noise_for_sampling = torch.randn_like(x_t)

        is_not_last_step_mask = (
            (t_long != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )

        sampled_x_t_minus_1 = (
            model_mean
            + is_not_last_step_mask
            * torch.exp(0.5 * model_log_variance)
            * noise_for_sampling
        )

        return sampled_x_t_minus_1

    def sample(
        self, model: nn.Module, num_images: int, image_shape: tuple, batch_size: int = 4
    ) -> torch.Tensor:
        all_generated_images = []
        model.eval()

        num_batches_to_generate = math.ceil(num_images / batch_size)

        for i in range(num_batches_to_generate):
            current_batch_size = min(batch_size, num_images - (i * batch_size))
            if current_batch_size == 0:
                break

            x_t = torch.randn(
                (current_batch_size, *image_shape),
                device=self.device,
                dtype=self.betas.dtype,
            )
            for t_step in reversed(range(self.num_timesteps)):
                t_tensor = torch.full(
                    (current_batch_size,), t_step, device=self.device, dtype=torch.long
                )

                x_t = self.p_sample(model, x_t, t_tensor)

            all_generated_images.append(x_t.cpu())

        final_images = torch.cat(all_generated_images, dim=0)
        return final_images
