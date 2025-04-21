import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def show_image_grid(samples, nrow = 4):
    """
    Displays a grid of generated image samples using matplotlib.
    Displays the first nrow² samples in the batch.

    Args:
        samples (torch.Tensor): Tensor of shape (N, 1, H, W) containing image samples.
        nrow (int): Number of rows (and columns) in the grid. The grid will display nrow * nrow images.
    """
    samples = samples.cpu().clamp(0.0, 1.0)
    fig, axes = plt.subplots(nrow, nrow, figsize=(6, 6))
    for i in range(nrow):
        for j in range(nrow):
            axes[i, j].imshow(samples[i * nrow + j, 0], cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def show_loss_curves(train_losses, val_losses, title='Training & Validation Loss'):
    """
    Plots training and validation loss curves over epochs.

    Args:
        train_losses (list or array-like): List of training loss values recorded at each epoch.
        val_losses (list or array-like): List of validation loss values recorded at each epoch.
        title (str): Title of the plot (default: "Training & Validation Loss").
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_sigma_t_curves(sigmas=[0.1, 1.01, 2, 5], t_range=(0, 1), num_points=200):
    """
    Plots $\sigma^t$ for multiple sigma values over a given interval.

    Args:
        sigmas (list of float): Sigma values to plot. Default is [0.1, 1.01, 2, 5].
        t_range (tuple): Interval (start, end) for t. Default is (0, 1).
        num_points (int): Number of points to sample in the interval. Default is 200.
    """
    t_vals = np.linspace(t_range[0], t_range[1], num_points)

    plt.figure(figsize=(8, 5))
    for sigma in sigmas:
        y_vals = sigma ** t_vals
        plt.plot(t_vals, y_vals, label=f"$\\sigma={sigma}$")

    plt.title("$\sigma^t$ as a function of $t$")
    plt.xlabel("$t$")
    plt.ylabel("$\sigma^t$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_VE_variance_curves(sigmas=None, t_range=(0, 1), num_points=200):
    """
    Plots the VE SDE variance function: 
    $\lambda(t) = (sigma^{2t} - 1)(2 log(sigma)) for multiple sigma values.

    Args:
        sigmas (list of float): Sigma values to plot. Defaults to [0.1, 1.01, 2, 10, 20].
        t_range (tuple): Interval (start, end) for t. Default is (0, 1).
        num_points (int): Number of points to sample in the interval. Default is 200.
    """
    if sigmas is None:
        sigmas = [0.1, 1.01, 2, 10, 20]

    t = np.linspace(t_range[0], t_range[1], num_points)

    def lambda_t(t, sigma):
        return (sigma**(2*t) - 1) / (2 * np.log(sigma))

    plt.figure(figsize=(8, 5))
    for sigma in sigmas:
        plt.plot(t, lambda_t(t, sigma), label=f"$\\sigma$ = {sigma}")
    
    plt.xlabel("$t$")
    plt.ylabel("Variance of $\\mathbf{x}_t$")
    plt.title("Variance evolution for VE SDE with different initial variances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_lambda1_VE_vs_sigma(sigmas_range=(2, 100), num_points=500):
    """
    Plots the final VE SDE variance 
    $\lambda(1) = (\sigma^2 - 1)/(2 log(sigma))$ as a function of sigma.

    Args:
        sigmas_range (tuple): Range (min, max) of sigma values. Default is (2, 100).
        num_points (int): Number of sigma values to sample. Default is 500.
    """
    def lambda_squared_final(sigma):
        return (sigma**2 - 1) / (2 * np.log(sigma))

    sigmas = np.linspace(sigmas_range[0], sigmas_range[1], num_points)
    variances = lambda_squared_final(sigmas)

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, variances)
    plt.axvline(25, color='gray', linestyle='--', label=r'$\sigma=25$')
    plt.axvline(83.28, color='gray', linestyle=':', label=r'$\sigma=83.28$')
    plt.title(r'Final Variance $\lambda(1)$ as a Function of $\sigma$')
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$\lambda(1)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_VP_variance_curves(var0_list=None, t_range=(0, 1), num_points=200):
    """
    Plots the VP SDE variance over time: Var[x_t] = e^{-2t} * Var[x_0] + (1 - e^{-2t})
    for different initial variances.

    Args:
        var0_list (list of float): Initial variances $\mathrm{Var}[x_0]$. 
            Defaults to [0.01, 0.5, 1.0, 2.0, 10.0].
        t_range (tuple): Time interval (start, end). Default is (0, 1).
        num_points (int): Number of points to evaluate in the interval. Default is 200.
    """
    if var0_list is None:
        var0_list = [0.01, 0.5, 1.0, 2.0, 10.0]

    t = np.linspace(t_range[0], t_range[1], num_points)

    plt.figure(figsize=(8, 5))
    for var0 in var0_list:
        var_xt = np.exp(-2 * t) * var0 + (1 - np.exp(-2 * t))
        plt.plot(t, var_xt, label=f"$\\mathrm{{Var}}(x_0)$ = {var0}")
    
    plt.axhline(1.0, color='gray', linestyle='--', label="Preserved Variance = 1")
    plt.xlabel("$t$")
    plt.ylabel("Variance of $\\mathbf{x}_t$")
    plt.title("Variance evolution for VP SDE with different initial variances")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_random_time_series(X_data, x0=0, title="Data", figsize=(12, 4), n_plot=4):
    """
    Plot random univariate time series from real data.
    Each time series is prepended with an initial value `x0` at time t=0.

    Args:
        X_data (np.array): Array of shape (num_samples, sequence_length), containing real time series data.
        x0 (float): Initial value at t=0 to prepend to each time series for visualization.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure in inches (width, height).
        n_plot (int): Number of random time series to plot.
    """
    N = X_data.shape[-1]
    x_d = np.zeros((X_data.shape[0], N + 1))
    x_d[:, 0] = x0
    x_d[:, 1:] = X_data
    X_data = x_d

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for _ in range(n_plot):
        j = np.random.randint(len(X_data))
        ax.plot(X_data[j], linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.show()


def show_real_vs_sampled_ts(X_data, X_sbts, x0=0):
    """
    Plot 4 unique univariate time series from real and generated data.

    This function visualizes and compares 4 time series sampled from:
    - the original dataset (X_data)
    - the generated samples (X_sbts)

    Each time series is prepended with an initial value `x0` at time t=0.

    Args:
        X_data (np.array): Array of shape (num_samples, sequence_length), containing real time series data.
        X_sbts (np.array): Array of shape (num_samples, sequence_length), containing generated time series samples.
        x0 (float): Initial value at t=0 to prepend to each time series for visualization (default: 0).
    """
    N = X_data.shape[-1]
    x_d, x_s = np.zeros((X_data.shape[0], N + 1)), np.zeros((X_sbts.shape[0], N + 1))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts
    X_data, X_sbts = x_d, x_s

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    n = min(4, len(X_data), len(X_sbts))
    idx_data = np.random.choice(len(X_data), size=n, replace=False)
    idx_sbts = np.random.choice(len(X_sbts), size=n, replace=False)

    for j1, j2 in zip(idx_data, idx_sbts):
        ax[0].plot(X_data[j1], linewidth=1.5)
        ax[1].plot(X_sbts[j2], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SNTS')
    ax[1].tick_params(axis='both', which='major', labelsize=13)
    plt.show()


def show_ts_and_embedding(ts: torch.Tensor, ts_img: torch.Tensor, title: str = "Original Time Series"):
    """
    Plot a univariate time series and its delay embedding side by side.

    Args:
        ts (torch.Tensor): Time series of shape (L,), (1, L), or (L, 1)
        ts_img (torch.Tensor): Delay embedding image of shape (1, H, W)

    Returns:
        None
    """
    ts_np = ts.detach().cpu().view(-1).numpy()       # shape (L,)
    img_np = ts_img[0].detach().cpu().numpy()        # shape (H, W)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # Time series
    axes[0].plot(ts_np, color='tab:blue')
    axes[0].set_title(title)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].grid(True)

    # Delay embedding image
    im = axes[1].imshow(img_np, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[1].set_title("Delay Embedding Image")
    axes[1].set_xlabel("Column (lag)")
    axes[1].set_ylabel("Row (embedding)")
    fig.colorbar(im, ax=axes[1], label="Amplitude")

    fig.subplots_adjust(wspace=0.35)
    plt.show()


def show_noisy_time_series(x0, sigmas=[0.01, 0.05, 0.1, 0.2, 0.5], n_series=5):
    """
    Visualize time series with different levels of additive Gaussian noise.

    Args:
        x0 (np.ndarray): Array of shape (n_series_total, sequence_length), original time series.
        sigmas (list of float): List of standard deviations for Gaussian noise.
        n_series (int): Number of time series to display.
    """
    x0 = x0[:n_series]  # Select first n_series for visualization
    fig, axs = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4), sharey=True)

    for i, sigma in enumerate(sigmas):
        noise = np.random.randn(*x0.shape) * sigma
        xt = x0 + noise
        axs[i].plot(xt.T, alpha=0.8)
        axs[i].set_title(f"$\\sigma$ = {sigma:.2f}")
        axs[i].set_xlabel("Time")
        if i == 0:
            axs[i].set_ylabel("Value")

    fig.suptitle("Additive Gaussian Noise on Time Series", fontsize=14)
    plt.tight_layout()
    plt.show()