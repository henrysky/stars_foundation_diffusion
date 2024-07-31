from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .ddpm import ConditionalDiffusionModel


def get_initializer(
    initializer: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if initializer is None:
        initializer = "xavier_uniform_"
    if isinstance(initializer, str):
        initializer = getattr(torch.nn.init, initializer)
    if callable(initializer):
        return initializer
    else:
        raise TypeError("Initializer function must be callable")


def get_activation(
    activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation is None:
        activation = "relu"
    if isinstance(activation, str):
        activation = getattr(F, activation)
    if callable(activation):
        return activation
    else:
        raise TypeError("Activation function must be callable")


class NonLinearEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = "uniform_",
        bias_initializer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "elu",
        use_bias: bool = True,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Non-linear embedding layer

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            embeddings_initializer (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): initializer for embeddings. Defaults to "uniform_".
            bias_initializer (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): initializer for bias. Defaults to None.
            activation (Union[str, Callable], optional): activation function. Defaults to "elu".
            use_bias (bool, optional): whether to use bias. Defaults to True.
            device (Union[str, torch.device], optional): device to run PyTorch on. Defaults to "cpu".
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = get_initializer(embeddings_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.activation_fn = get_activation(activation)
        self.use_bias = use_bias

        # always reserve token 0 as special padding token
        self.padding_idx = 0

        self.device = device
        self.dtype = dtype

        self.embeddings = Parameter(
            torch.empty((self.input_dim, self.output_dim), **self.factory_kwargs)
        )
        self.bias = Parameter(
            torch.empty((self.input_dim, self.output_dim), **self.factory_kwargs)
        )
        self.reset_parameters()

    @property
    def factory_kwargs(self) -> dict:
        return {"device": self.device, "dtype": self.dtype}

    def reset_parameters(self):
        self.embeddings_initializer(self.embeddings)
        if self.use_bias:
            self.bias_initializer(self.bias)
        else:
            with torch.no_grad():
                torch.nn.init.zeros_(self.bias)

        with torch.no_grad():
            self.embeddings[self.padding_idx].fill_(0)
            self.bias[self.padding_idx].fill_(0)

    def forward(self, input_tokens: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        idx = input_tokens
        out = F.embedding(idx, self.embeddings)
        bias = F.embedding(idx, self.bias)
        return self.activation_fn(out * inputs + bias)

    def get_embedding(self, input_tokens: torch.Tensor) -> torch.Tensor:
        out = F.embedding(input_tokens, self.embeddings)
        return out


class StellarPerceptronTorchModelwDDPM(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module,
        embedding_dim: int,
        num_heads: int,
        dense_num: int,
        dropout_rate: float,
        activation: Callable[[torch.Tensor], torch.Tensor],
        num_layers: int = 2,
        diffusion_dense_num: int = 128,
        diffusion_n_layers: int = 3,
        diffusion_num_steps: int = 100,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        built: bool = False,  # do not use this arguement, it is for internal use only
    ):
        """
        Model with transformer encoder layers and diffusion model as head

        Args:
            embedding_layer (nn.Module): embedding layer
            embedding_dim (int): embedding dimension
            num_heads (int): number of heads in multi-head attention
            dense_num (int): number of neurons in the dense layer
            dropout_rate (float): dropout rate
            activation (Callable[[torch.Tensor], torch.Tensor]): activation function
            num_layers (int, optional): number of transformer encoder layers. Defaults to 2.
            diffusion_dense_num (int, optional): number of neurons in the diffusion dense layer. Defaults to 128.
            diffusion_n_layers (int, optional): number of layers in the diffusion model. Defaults to 3.
            diffusion_num_steps (int, optional): number of diffusion steps. Defaults to 100.
            device (Union[str, torch.device], optional): device to run PyTorch on. Defaults to "cpu".
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.embedding_layer = embedding_layer
        self.activation = get_activation(activation)
        self._built = built

        self.encoder_transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    nhead=num_heads,
                    dim_feedforward=dense_num,
                    d_model=embedding_dim,
                    dropout=dropout_rate,
                    activation=activation,
                    batch_first=True,
                    **self.factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.diffusion_head = ConditionalDiffusionModel(
            dim=1,
            cond_dim=embedding_dim,
            dense_num=diffusion_dense_num,
            n_layers=diffusion_n_layers,
            num_steps=diffusion_num_steps,
            built=self._built,
            **self.factory_kwargs,
        )

    @property
    def factory_kwargs(self) -> dict:
        return {"device": self.device, "dtype": self.dtype}

    def get_config(self) -> dict:
        return {
            "embedding_layer": self.embedding_layer,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "dense_num": self.dense_num,
            "activation": self.activation,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        }

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_token_tensor: torch.Tensor,
        output_token_tensor: torch.Tensor,
    ) -> torch.Tensor:
        input_embedded = self.embedding_layer(input_token_tensor, input_tensor)
        output_embedding = self.embedding_layer.get_embedding(output_token_tensor)
        concat_tensor = torch.cat([output_embedding, input_embedded], dim=1)
        concat_token = torch.cat([output_token_tensor, input_token_tensor], dim=1)
        padding_mask = torch.eq(concat_token, torch.zeros_like(concat_token))

        for layer in self.encoder_transformer_blocks:
            concat_tensor = layer(src=concat_tensor, src_key_padding_mask=padding_mask)
        perception_first_pos = concat_tensor[:, 0]
        return perception_first_pos

    def get_loss(
        self,
        input_tensor: torch.Tensor,
        input_token_tensor: torch.Tensor,
        y_token_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> torch.Tensor:
        out = self(input_tensor, input_token_tensor, y_token_tensor)

        # diffusion loss
        diffusion_loss = self.diffusion_head.noise_estimation_loss(y_tensor, out)

        return diffusion_loss
