import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
    This module maps scalar inputs (e.g., time steps) to a higher-dimensional 
    space using fixed random Fourier features, based on a Gaussian distribution.
    """

    def __init__(self, embed_dim, scale = 30.):
        """
        Initializes the GaussianFourierProjection module.

        Args:
            embed_dim (int): Output dimension of the encoding.
            scale (float, optional): Standard deviation used to scale the sampled Gaussian 
                frequencies. Controls the frequency range of the encodings.
        """
        super().__init__()
        # Random sampling of weights. These weights are fixed during optimization.
        random_weights = torch.randn(embed_dim // 2)
        scaled_weights = random_weights * scale
        self.W = nn.Parameter(scaled_weights, requires_grad=False)

    def forward(self, x):
        """
        Applies the random Fourier feature encoding to the input tensor.

        Args:
            x (torch.Tensor): 1D tensor of shape (batch_size,) containing scalar 
                values (e.g., time steps) to encode.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, embed_dim).
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        return torch.cat([x_sin, x_cos], dim=-1)


class Dense(nn.Module):
    """
    A fully connected (linear) layer that reshapes its output to match
    a 2D feature map format, typically used before upsampling in CNNs.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initializes the Dense layer.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features (channels).
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Applies a linear transformation to the input and reshapes it to 
        a 4D tensor with singleton spatial dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim, 1, 1),
                          suitable for use as a feature map in convolutional models.
        """
        out = self.dense(x)                # (batch_size, output_dim)
        out_reshaped = out[..., None, None]  # (batch_size, output_dim, 1, 1)
        return out_reshaped


class ScoreNet(nn.Module):
    """
    A time-dependent score-based model built upon a U-Net architecture.

    This model takes as input a noisy data sample `x(t)` and a time step `t`,
    and learn the score (i.e., the gradient of the log-density) using a
    deep neural network conditioned on time via Fourier embeddings.
    """

    def __init__(
        self,
        perturbation_kernel_std,
        channels = [32, 64, 128, 256],
        embed_dim = 256,
        padding=0
    ):
        """
        Initializes the ScoreNet model.

        Args:
            marginal_prob_std (Callable): A function that takes a tensor of time steps `t`
                and returns the standard deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels (List[int], optional): List of channel sizes for each stage of the U-Net.
            embed_dim (int, optional): Dimensionality of the Fourier feature time embeddings.
        """
        super().__init__()

        self.perturbation_kernel_std = perturbation_kernel_std

        # Gaussian random feature embedding of time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding path (downsampling)
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=padding, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=padding, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=padding, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=padding, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoding path (upsampling)
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=padding, output_padding=padding, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2]*2, channels[1], 3, stride=2, padding=padding, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1]*2, channels[0], 3, stride=2, padding=padding, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0]*2, 1, 3, stride=1, padding=padding)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        """
        Forward pass of the time-dependent score model.

        Args:
            x (torch.Tensor): Noisy input data of shape (batch_size, 1, H, W).
            t (torch.Tensor): Time steps of shape (batch_size,).

        Returns:
            torch.Tensor: Predicted score function ∇ₓ log p_{t}(x), same shape as `x`.
        """
        # Compute time embedding
        embed = self.act(self.embed(t))  # (batch_size, embed_dim)

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.perturbation_kernel_std(t)[:, None, None, None]
        return h