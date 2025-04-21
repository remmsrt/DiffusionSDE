import torch
from tqdm.notebook import tqdm


def euler_sampler_VE(
    score_model,
    perturbation_kernel_std,
    diffusion_coeff,
    batch_size=16,
    num_steps=500,
    device='cpu',
    eps=1e-3,
    show_progress=True,
    x_mean=0,
    input_size=(1, 28, 28)
):
    """
    Generates samples from a score-based model using the Euler-Maruyama method,
    which approximates the reverse-time SDE: dx = [g(t)^2 * score] dt + g(t) d𝑤̄.

    Args:
        score_model (torch.nn.Module): Time-dependent score network s_theta(x, t).
        perturbation_kernel_std (Callable[[torch.Tensor], torch.Tensor]): Function sigma(t) giving std of the forward kernel.
        diffusion_coeff (Callable[[torch.Tensor], torch.Tensor]): Function g(t) giving diffusion coefficient of the SDE.
        batch_size (int): Number of samples to generate.
        num_steps (int): Number of time steps for discretization (Euler steps).
        device (str): 'cuda' or 'cpu'.
        eps (float): Final time value (sampling ends at t = eps).
        show_progress (bool): Whether to show a progress bar.

    Returns:
        torch.Tensor: Samples from p_0, of shape (batch_size, 1, 28, 28).
    """
    # Initial time t = 1 for all samples
    t_init = torch.ones(batch_size, device=device)

    # Sample initial noise from prior: x_T ~ N(0, σ(t)^2 I)
    x = torch.randn(batch_size, *input_size, device=device) * perturbation_kernel_std(t_init)[:, None, None, None] + x_mean

    # Time discretization: from t = 1 to t = eps
    time_steps = torch.linspace(1., eps, num_steps, device=device) 
    step_size = time_steps[0] - time_steps[1] # dt

    iterator = tqdm(time_steps) if show_progress else time_steps

    with torch.no_grad():
        for t in iterator:
            batch_t = torch.ones(batch_size, device=device) * t
            g = diffusion_coeff(batch_t)
            score = score_model(x, batch_t)

            # Euler-Maruyama update: x ← x + drift + noise
            drift = (g ** 2)[:, None, None, None] * score * step_size
            noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            x = x + drift + noise

    return x  # Final sample at t = eps (approximate p_0)


def euler_sampler_VP(
    score_model,
    diffusion_coeff,
    batch_size=16,
    num_steps=500,
    device='cpu',
    eps=1e-3,
    show_progress=True,
    input_size=(1, 28, 28)
):
    """
    Generates samples from a score-based model using the Euler-Maruyama method
    for the reverse-time VP SDE: dx = [x + g(t)^2 * score] dt + g(t) dw.

    Args:
        score_model (torch.nn.Module): Time-dependent score network s_theta(x, t).
        diffusion_coeff (Callable[[torch.Tensor], torch.Tensor]): Function g(t) giving diffusion coefficient of the SDE.
        batch_size (int): Number of samples to generate.
        num_steps (int): Number of Euler time steps.
        device (str): 'cuda' or 'cpu'.
        eps (float): Final time value (sampling ends at t = eps).
        show_progress (bool): Whether to show a progress bar.

    Returns:
        torch.Tensor: Samples from p_0, of shape (batch_size, 1, 28, 28).
    """
    # Sample from standard normal prior: x_T ~ N(0, I)
    x = torch.randn(batch_size, *input_size, device=device)

    # Time discretization: t from 1 → eps
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1] # dt

    iterator = tqdm(time_steps) if show_progress else time_steps

    with torch.no_grad():
        for t in iterator:
            batch_t = torch.ones(batch_size, device=device) * t
            g = diffusion_coeff(batch_t)
            score = score_model(x, batch_t)

            # Euler-Maruyama update: x ← x + drift + noise
            drift = (x + (g ** 2)[:, None, None, None] * score) * step_size
            noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            x = x + drift + noise

    return x  # Final sample at t = eps (approximate p_0)