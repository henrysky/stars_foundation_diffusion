"""Denosing diffusion probabilistic model"""

from typing import Callable, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        t: int,
        cond_dim: int = 0,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Conditional linear layer used in diffusion model

        Args:
            dim_in (int): input dimension
            dim_out (int): output dimension, also the time embedding dimension
            t (int): total number of time step tokens
            cond_dim (int, optional): condition dimension, by default 0 (no condition)
            device (Union[str, torch.device], optional): device to run the model, by default "cpu"
            dtype (torch.dtype, optional): data type of the model, by default torch.float32
        """
        super(ConditionalLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.t = t
        self.cond_dim = cond_dim
        self.factory_kwargs = {"device": device, "dtype": dtype}
        if self.cond_dim < 0:
            raise ValueError("cond_dim must be non-negative")
        else:
            if self.cond_dim != self.dim_out and self.cond_dim != 0:
                # since the condition is added to the time embedding, it must have the same dimension as the output so we are mapping it to the same dimension
                warnings.warn(
                    f"In ideal case, cond_dim equals to dim_out. Now cond_dim ({self.cond_dim}) is linearly mapped to dim_out ({self.dim_out}) so you have additional trainable parameters."
                )
                self.cond_lin = nn.Linear(
                    self.cond_dim,
                    self.dim_out,
                    **self.factory_kwargs,
                )
            else:
                self.cond_lin = torch.nn.Identity()
        self.lin = nn.Linear(
            self.dim_in,
            self.dim_out,
            **self.factory_kwargs,
        )
        # time embedding
        self.embed = nn.Embedding(self.t, self.dim_out, **self.factory_kwargs)
        self.embed.weight.data.uniform_()  # uniform initialization of the time embedding

        # forward function depends on whether a condition is provided
        if self.cond_dim > 0:
            self.forward = self._forward_cond
        else:
            self.forward = self._forward_simple

        # if no condition is provided, use this zero tensor to NOT modify the time embedding
        self.a_zero = torch.zeros(1, **self.factory_kwargs)

    def time_forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor
            t (torch.Tensor): time token
            cond (torch.Tensor): condition tensor
        """
        gamma = self.embed(t) + self.cond_lin(cond)
        out = gamma.view(-1, self.dim_out) + x
        return out

    def _forward_simple(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor
            t (torch.Tensor): time token
        """
        out = self.lin(x)
        out = self.time_forward(out, t, self.a_zero)
        return out

    def _forward_cond(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor
            t (torch.Tensor): time token
            cond (torch.Tensor): condition tensor
        """
        out = self.lin(x)
        out = self.time_forward(out, t, cond)
        return out


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int = 0,
        num_steps: int = 100,
        dense_num: int = 128,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        beta_start: float = 1.0e-5,
        beta_end: float = 1.0e-1,
        schedule: str = "sigmoid",
    ):
        """
        Conditional denoising diffusion probabilistic model, Ho et al 2020 (https://arxiv.org/abs/2006.11239)

        Args:
            dim (int): input dimension
            cond_dim (int, optional): condition dimension, by default 0 (no condition)
            num_steps (int, optional): number of diffusion steps, by default 100
            dense_num (int, optional): number of neurons in the dense layers, by default 128
            activation (Union[str, Callable[[torch.Tensor], torch.Tensor]], optional): activation function, by default "gelu"
            device (Union[str, torch.device], optional): device to run the model, by default "cpu"
            dtype (torch.dtype, optional): data type of the model, by default torch.float32
            beta_start (float, optional): starting value of beta, by default 1.0e-5
            beta_end (float, optional): ending value of beta, by default 1.0e-1
            schedule (str, optional): beta schedule, by default "sigmoid"
        """
        super().__init__()
        self.num_steps = num_steps
        self.dim = dim
        self.cond_dim = cond_dim
        self.dense_num = dense_num
        self.activation = self.get_activation(activation)
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = ConditionalLinear(
            self.dim,
            self.dense_num,
            self.num_steps,
            self.cond_dim,
            **self.factory_kwargs,
        )
        self.lin2 = ConditionalLinear(
            self.dense_num,
            self.dense_num,
            self.num_steps,
            self.cond_dim,
            **self.factory_kwargs,
        )
        self.lin3 = ConditionalLinear(
            self.dense_num,
            self.dense_num,
            self.num_steps,
            self.cond_dim,
            **self.factory_kwargs,
        )
        self.lin4 = nn.Linear(self.dense_num, self.dim, **self.factory_kwargs)

        # ============ diffusion parameters ============
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.betas = self.make_beta_schedule(schedule=self.schedule)

        self.alphas = 1.0 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat(
            [torch.tensor([1], **self.factory_kwargs), self.alphas_prod[:-1]], 0
        )
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.reversed_t_array = torch.arange(
            start=self.num_steps - 1,
            end=-1,
            step=-1,
            device=self.factory_kwargs["device"],
        ).view(self.num_steps, 1)

        if self.cond_dim > 0:
            self.forward = self._forward_cond
        else:
            self.forward = self._forward_simple

    def get_config(self) -> dict:
        """
        Function to return the configuration of the model
        """
        return {
            "dim": self.dim,
            "cond_dim": self.cond_dim,
            "num_steps": self.num_steps,
            "dense_num": self.dense_num,
            "activation": self.activation.__name__,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "schedule": self.schedule,
        }

    def get_parameters_sum(self) -> int:
        """
        Function to count the tortal number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def get_activation(
        self, activation: Union[str, Callable]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation is None:
            activation = "relu"
        if isinstance(activation, str):
            activation = getattr(F, activation)
        if callable(activation):
            return activation
        else:
            raise ValueError(
                "Activation function must be callable or string of PyTorch activation function"
            )

    def _forward_simple(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.activation(self.lin1(x, t))
        x = self.activation(self.lin2(x, t))
        x = self.activation(self.lin3(x, t))
        return self.lin4(x)

    def _forward_cond(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.activation(self.lin1(x, t, cond))
        x = self.activation(self.lin2(x, t, cond))
        x = self.activation(self.lin3(x, t, cond))
        return self.lin4(x)

    def p_sample(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None
    ):
        """
        Perform a single denoising step from t+1 to t

        Args:
            x (torch.Tensor): input tensor
            t (torch.Tensor): time token
            cond (torch.Tensor, optional): condition tensor
        """
        # Factor to the model output
        eps_factor = (1 - self.extract(self.alphas, t, x)) / self.extract(
            self.one_minus_alphas_bar_sqrt, t, x
        )
        eps_theta = self(x, t, cond)
        # Final values
        mean = (1 / self.extract(self.alphas, t, x).sqrt()) * (
            x - (eps_factor * eps_theta)
        )
        # Generate z
        z = torch.randn_like(x)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, x).sqrt()
        sample = mean + sigma_t * z
        return sample

    def p_sample_loop(
        self,
        size: int,
        cond: Optional[torch.Tensor] = None,
        return_steps: bool = False,
    ) -> torch.Tensor:
        """
        Perform the whole denoising step from pure noise to the final output

        Args:
            size int: size of samples to generate
            cond (torch.Tensor, optional): condition tensor, by default None
            return_steps (bool, optional): return intermediate steps, by default False
        """
        cur_x = torch.randn(size, self.dim, **self.factory_kwargs)
        if return_steps:
            x_seq = torch.zeros((self.num_steps, size, self.dim), **self.factory_kwargs)
        for idx, i in enumerate(self.reversed_t_array):
            cur_x = self.p_sample(
                cur_x,
                i,
                cond,
            )
            if return_steps:
                x_seq[idx] = cur_x
        return cur_x if not return_steps else x_seq

    def noise_estimation_loss(
        self,
        x_0: torch.Tensor,
        cond_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Noise estimation loss

        Args:
            x_0 (torch.Tensor): input tensor
            cond_x (torch.Tensor, optional): condition tensor, by default None
        """
        batch_size = x_0.shape[0]
        # Select a random step for each example
        t = torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size // 2 + 1,),
            device=self.factory_kwargs["device"],
        )
        t = torch.cat([t, self.num_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, x_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        e = torch.randn_like(x_0)
        # model input
        x = x_0 * a + e * am1
        output = self(x, t, cond_x)
        loss = torch.nn.functional.mse_loss(output, e)
        return loss

    def make_beta_schedule(self, schedule: str = "linear") -> torch.Tensor:
        if schedule == "linear":
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_steps, **self.factory_kwargs
            )
        elif schedule == "quad":
            betas = (
                torch.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_steps,
                    **self.factory_kwargs,
                )
                ** 2
            )
        elif schedule == "cosine":

            def cosine_fn(t):
                return torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2

            betas = torch.zeros(self.num_steps, **self.factory_kwargs)
            max_beta = torch.tensor(0.999, **self.factory_kwargs)
            for i in range(self.num_steps):
                t1 = torch.tensor(i / self.num_steps, **self.factory_kwargs)
                t2 = torch.tensor((i + 1) / self.num_steps, **self.factory_kwargs)
                betas[i] = torch.min(1 - cosine_fn(t2) / cosine_fn(t1), max_beta)
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.num_steps, **self.factory_kwargs)
            betas = (
                torch.sigmoid(betas) * (self.beta_end - self.beta_start)
                + self.beta_start
            )
        else:
            raise ValueError("Unknown beta schedule")
        return betas

    def extract(
        self, input: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the value of the input tensor at the given time token
        """
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def q_x(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform a single noising step from t to t+1

        Args:
            x_0 (torch.Tensor): input tensor
            t (torch.Tensor): time token
        """
        noise = torch.randn_like(x_0)
        alphas_t = self.extract(self.alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = self.extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        return alphas_t * x_0 + alphas_1_m_t * noise
