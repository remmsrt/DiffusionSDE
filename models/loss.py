import torch


def loss_VE(
    model, 
    x, 
    perturbation_kernel_std, 
    eps = 1e-5
):
    """
    Computes the denoising score matching (DSM) loss for the VE SDE: dx = sigma^t dw.

    Args:
        model (torch.nn.Module): Score network taking noisy input x_t and time t.
        x (torch.Tensor): Input batch of shape (batch_size, 1, H, W).
        perturbation_kernel_std (Callable[[torch.Tensor], torch.Tensor]): Function sigma(t) giving std of the perturbation kernel.
        eps (float): Small constant to avoid numerical issues near t=0 (default: 1e-5).

    Returns:
        torch.Tensor: Scalar loss value (DSM).
    """
    batch_size = x.shape[0]
    device = x.device

    # Sample random times t ~ Uniform(eps, 1)
    t = torch.rand(batch_size, device=device) * (1. - eps) + eps 

    # Add Gaussian noise: x_t = x + sigma(t) * z
    std = perturbation_kernel_std(t)
    z = torch.randn_like(x)
    x_t = x + std[:, None, None, None] * z

    # Predict score s_theta(x_t, t)
    score = model(x_t, t)  

    # DSM Loss: E[|| sqrt(lambda(t)) * score + z ||^2]
    loss = torch.mean(torch.sum((std[:, None, None, None] * score + z) ** 2, dim=(1, 2, 3))) 
    return loss


def loss_VP(
    model, 
    x, 
    perturbation_kernel_std, 
    eps = 1e-5
):
    """
    Computes the denoising score matching (DSM) loss for the VP SDE: dx = -x + sqrt(2)dw.

    Args:
        model (torch.nn.Module): Score network taking noisy input x_t and time t.
        x (torch.Tensor): Input batch of shape (batch_size, 1, H, W).
        perturbation_kernel_std (Callable[[torch.Tensor], torch.Tensor]): Function sigma(t) giving std of the perturbation kernel.
        eps (float): Small constant to avoid numerical issues near t=0 (default: 1e-5).

    Returns:
        torch.Tensor: Scalar loss value (DSM).
    """
    batch_size = x.shape[0]
    device = x.device

    # Sample random times t ~ Uniform(eps, 1)
    t = torch.rand(batch_size, device=device) * (1. - eps) + eps

    # Add Gaussian noise: x_t = exp(-t) * x + sigma(t) * z
    std = perturbation_kernel_std(t)
    exp_neg_t = torch.exp(-t)
    z = torch.randn_like(x)
    x_t = exp_neg_t[:, None, None, None] * x + std[:, None, None, None] * z

    # Predict score s_theta(x_t, t)
    score = model(x_t, t)

    # DSM Loss: E[|| sqrt(lambda(t)) * score + z ||^2]
    loss = torch.mean(torch.sum((std[:, None, None, None] * score + z) ** 2, dim=(1, 2, 3)))
    return loss