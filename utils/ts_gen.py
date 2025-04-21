import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

class GenerateData:
    def __init__(self, M):
        """
        Initialize the data generator.

        Args:
            M (int): Number of time series to generate.
        """

        self.M = M

    def generate_OU(self, theta_range, mu_range, sigma_range, N, dt=1 / 252, x0=1):
        """
        Generate univariate time series from an Ornstein-Uhlenbeck (OU) process.

        Args:
            theta_range (list of float): Range [min, max] for the mean-reversion rate theta.
            mu_range (list of float): Range [min, max] for the long-term mean mu.
            sigma_range (list of float): Range [min, max] for the volatility sigma.
            N (int): Length of each time series.
            dt (float): Time step size (default: 1/252 for daily data).
            x0 (float): Initial value of the process at t=0.

        Returns:
            np.ndarray: Array of shape (M, N + 1) with OU time series.
        """

        def simulate(theta, mu, sigma):
            X = np.zeros(N + 1)
            X[0] = x0
            for t in range(1, N + 1):
                mu_t = X[t - 1] * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
                sigma_t = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
                X[t] = mu_t + np.sqrt(sigma_t) * np.random.normal(0, 1)
            return X

        thetas = np.random.uniform(theta_range[0], theta_range[1], self.M)
        mus = np.random.uniform(mu_range[0], mu_range[1], self.M)
        sigmas = np.random.uniform(sigma_range[0], sigma_range[1], self.M)
        return np.array([simulate(thetas[i], mus[i], sigmas[i]) for i in range(self.M)])
    

def generate_OU_dataset(M_target, N, theta_range, mu_range, sigma_range, buffer=2):
    """
    Generate a dataset of Ornstein-Uhlenbeck time series trajectories with strictly positive values.

    Args:
        M_target (int): Number of valid (strictly positive) trajectories to return.
        N (int): Length of each trajectory.
        theta_range (tuple): Range of values for the OU parameter theta (mean reversion rate).
        mu_range (tuple): Range of values for the OU parameter mu (long-term mean).
        sigma_range (tuple): Range of values for the OU parameter sigma (volatility).
        buffer (int): Factor to over-generate trajectories (to increase the chance of getting valid ones).

    Returns:
        np.ndarray: Array of shape (M_target, N) containing OU trajectories with only positive values.
    """
    M_buffer = M_target * buffer
    Generate_data = GenerateData(M=M_buffer)
    X_all = Generate_data.generate_OU(theta_range, mu_range, sigma_range, N)
    X_filtered = X_all[np.all(X_all > 0, axis=1)] # keep only trajectories that are strictly positive at all time steps

    while len(X_filtered) < M_target: # keep generating until we reach the desired number of valid samples
        Generate_data = GenerateData(M=M_buffer)
        X_new = Generate_data.generate_OU(theta_range, mu_range, sigma_range, N)
        X_new_filtered = X_new[np.all(X_new > 0, axis=1)]
        X_filtered = np.concatenate([X_filtered, X_new_filtered], axis=0)

    return X_filtered[:M_target] # return only the first M_target valid samples


def MLE_OU_robust(params, X, dt):
    """
    Compute the MLE on Ornstein-Uhlenbeck data.
    
    This function evaluates how well a set of parameters (theta, mu, sigma)
    fits the observed univariate time series X assumed to follow an OU process.

    Args:
        params (list or tuple): Parameters [theta, mu, sigma] of the OU process.
        X (np.ndarray): Observed time series of shape (N,).
        dt (float): Time step between observations.

    Returns:
        float: Negative log-likelihood value (to be minimized).
    """
    theta, mu, sigma = params
    N = len(X)
    logL = 0

    exp_neg_theta_dt = np.exp(-theta * dt)
    one_minus_exp_neg_theta_dt = 1 - exp_neg_theta_dt
    sigma_eta2 = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

    for t in range(N - 1):
        mu_t = X[t] * exp_neg_theta_dt + mu * one_minus_exp_neg_theta_dt
        residual = X[t + 1] - mu_t
        logL += -0.5 * np.log(2 * np.pi * sigma_eta2) - (residual ** 2) / (2 * sigma_eta2)

    return -logL


def show_params_distrib_OU(X_data, X_sbts, dt, fix=False):
    """
    Estimate and visualize the distribution of Ornstein-Uhlenbeck parameters 
    (theta, mu, sigma) from real and generated time series.

    The function applies Maximum Likelihood Estimation (MLE) to each series
    in X_data and X_sbts, fits OU parameters, and then compares their 
    empirical distributions using kernel density estimates (KDE).

    Args:
        X_data (np.ndarray): Real time series of shape (num_series, time_steps).
        X_sbts (np.ndarray): Generated time series of shape (num_series, time_steps).
        dt (float): Time step between consecutive observations.
        fix (bool): If True, assumes the real parameters are fixed and plots them 
                    as vertical dashed lines. If False, assumes parameters are 
                    sampled from known distributions and plots their KDEs instead.

    Returns:
        None: Displays the distribution plots for the three OU parameters.
    """
    params_data = np.zeros((len(X_data), 3))
    for m in range(len(X_data)):
        params_init_data = [1, np.mean(X_data[m]), np.std(X_data[m])]
        result_data = minimize(MLE_OU_robust, np.array(params_init_data), args=(X_data[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_data[m] = result_data.x

    params_sbts = np.zeros((len(X_sbts), 3))
    for m in range(len(X_sbts)):
        params_init_sbts = [1, np.mean(X_sbts[m]), np.std(X_sbts[m])]
        result_sbts = minimize(MLE_OU_robust, np.array(params_init_sbts), args=(X_sbts[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_sbts[m] = result_sbts.x

    theta = np.random.uniform(.5, 2.5, 100000)
    mu = np.random.uniform(.5, 1.5, 100000)
    sigma = np.random.uniform(.1, .5, 100000)
    lines = [1.5, 1., 0.3]

    lower_bounds = np.percentile(params_data, 3, axis=0)
    upper_bounds = np.percentile(params_data, 97, axis=0)
    filtered_params_data = params_data[
        (params_data >= lower_bounds).all(axis=1) & (params_data <= upper_bounds).all(axis=1)]

    lower_bounds = np.percentile(params_sbts, 3, axis=0)
    upper_bounds = np.percentile(params_sbts, 97, axis=0)
    filtered_params_sbts = params_sbts[
        (params_sbts >= lower_bounds).all(axis=1) & (params_sbts <= upper_bounds).all(axis=1)]

    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    sns.kdeplot(ax=axs[0], data=params_data[:, 0], fill=True, label='Data')
    sns.kdeplot(ax=axs[0], data=params_sbts[:, 0], fill=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[0], data=theta, fill=True, label='Real')
    else:
        line_obj = axs[0].axvline(x=lines[0], color='black', linestyle='--',
                                  label='Real')
        axs[0].legend(handles=[line_obj])
    axs[0].set_title(r'Distribution of params $\theta$')
    axs[0].legend()

    sns.kdeplot(ax=axs[1], data=filtered_params_data[:, 1], fill=True, label='Data')
    sns.kdeplot(ax=axs[1], data=filtered_params_sbts[:, 1], fill=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[1], data=mu, fill=True, label='Real')
    else:
        line_obj = axs[1].axvline(x=lines[1], color='black', linestyle='--',
                                  label='Real')
        axs[1].legend(handles=[line_obj])
    axs[1].set_title(r'Distribution of params $\mu$')
    axs[1].legend()

    sns.kdeplot(ax=axs[2], data=params_data[:, 2], fill=True, label='Data')
    sns.kdeplot(ax=axs[2], data=params_sbts[:, 2], fill=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[2], data=sigma, fill=True, label='Real')
    else:
        line_obj = axs[2].axvline(x=lines[2], color='black', linestyle='--',
                                  label='Real')
        axs[2].legend(handles=[line_obj])
    axs[2].set_title(r'Distribution of params $\sigma$')
    axs[2].legend()

    fig.tight_layout()
    plt.show()