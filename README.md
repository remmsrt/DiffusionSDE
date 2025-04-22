# 📌 Score-Based Generative Modeling with SDEs

This repository contains an implementation of a score-based generative model using **stochastic differential equations (SDEs)**. It includes experiments on both image data (MNIST) and synthetic time series derived from Ornstein-Uhlenbeck processes.

Author: Rémi Surat  
Contact: remisurat@outlook.com

---

## 🧭 How to Read This Project

To understand and reproduce the methodology, I recommend the following order:

1. Start with [theory.ipynb](theory.ipynb) — It introduces the key mathematical concepts behind score-based diffusion modeling with SDEs.
2. Then read [demo_MNIST.ipynb](demo_MNIST.ipynb) — A practical example on the MNIST dataset.
3. Finally read [demo_TimeSeries.ipynb](demo_TimeSeries.ipynb) — An application on time series data.

---

## 📂 Project Structure

```
/DiffusionSDE
├── figures/                   # Nice figures for the notebooks
├── models/
│   ├── loss.py                # DSM losses for VE and VP SDEs 
│   ├── sampling.py            # Euler samplers for VE and VP SDEs
│   ├── score_net.py           # ScoreNet model (U-net + time embedding)
│   └── training.py            # Training loop
├── utils/
│   ├── ts_gen.py              # Ornstein Uhlenbeck time series generator
│   ├── imports.py             # Usefull library imports
│   ├── ts_to_img.py           # Embedding of time series
│   └── utils.py               # Seed, display functions
├── demo_MNIST.ipynb           # MNIST training & sampling demo
├── demo_TimeSeries.ipynb      # Time series training & sampling demo
└── theory.ipynb               # Theory behind score-based diffusion with SDEs          
```

---

## 🧪 Training & Sampling

Below is an example using a Variance Exploding (VE) SDE to train a score network on MNIST and generate samples. The same workflow applies to other datasets, such as Ornstein-Uhlenbeck time series, by changing the data and SDE configuration.

```python
# Train the model
train_score_model(
    model=score_model_MNIST_VE,                             # U-Net score network
    train_loader=train_loader_MNIST,                        # Dataloader for MNIST train set
    test_loader=test_loader_MNIST,                          # Dataloader for MNIST test set (validation)
    loss_fn=loss_VE,                                        # Denoising Score Matching loss (VE formulation)
    optimizer=optimizer_MNIST_VE,                           # Optimizer (e.g. Adam)
    perturbation_kernel_std=perturbation_kernel_std_VE,     # lambda(t) for the forward kernel
    device='cpu'                                            # or 'cuda' if GPU is available
)

# Sample new images
samples = euler_sampler_VE(
    score_model=score_model_MNIST_VE,                       # Trained score network
    perturbation_kernel_std=perturbation_kernel_std_VE,     # Same lambda(t) as training
    diffusion_coeff=diffusion_coeff_VE,                     # Diffusion coeff g(t) in the forward SDE
    input_size=(1, 28, 28),                                 # MNIST image shape
    device='cpu'
)
```

👉 See [demo_MNIST.ipynb](demo_MNIST.ipynb) and [demo_TimeSeries.ipynb](demo_TimeSeries.ipynb) for full examples.

---

## 📚 References

1. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).**  
   *Score-Based Generative Modeling through Stochastic Differential Equations*.  
   arXiv: [2011.13456](https://arxiv.org/abs/2011.13456)

2. **Song, Y., & Ermon, S. (2020).**  
   *Improved Techniques for Training Score-Based Generative Models*.  
   arXiv: [1907.05600](https://arxiv.org/abs/1907.05600)

3. **Hyvärinen, A. (2005).**  
   *Estimation of Non-Normalized Statistical Models by Score Matching*.  
   *Journal of Machine Learning Research*, 6, 695–709.  
   [https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

4. **Vincent, P. (2010).**  
   *A Connection Between Score Matching and Denoising Autoencoders*.  
   Technical Report, Université de Montréal.  
   [https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)

5. **Ho, J., Jain, A., & Abbeel, P. (2020).**  
   *Denoising Diffusion Probabilistic Models*.  
   *NeurIPS 2020*.  
   arXiv: [2006.11239](https://arxiv.org/abs/2006.11239)

6. **Anderson, B. D. O. (1982).**  
   *Reverse-time diffusion equation models*.  
   *Stochastic Processes and Their Applications*, 12(3), 313–326.  
   [https://core.ac.uk/download/pdf/82826666.pdf](https://core.ac.uk/download/pdf/82826666.pdf)

7. **Ronneberger, O., Fischer, P., & Brox, T. (2015).**  
   *U-Net: Convolutional Networks for Biomedical Image Segmentation*.  
   *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.  
   arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)

8. **Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., & Courville, A. (2019).**  
   *On the Spectral Bias of Deep Neural Networks*.  
   arXiv: [1806.08734](https://arxiv.org/abs/1806.08734)

9. **Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., & Ng, R. (2020).**  
   *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*.  
   *NeurIPS 2020*.  
   arXiv: [2006.10739](https://arxiv.org/abs/2006.10739)

10. **Azencot, O., Patel, D., Bao, C., & Haim, G. (2024).**  
    *Utilizing Image Transforms and Diffusion Models for Generative Modeling of Short and Long Time Series*.  
    arXiv: [2410.19538](https://arxiv.org/abs/2410.19538)