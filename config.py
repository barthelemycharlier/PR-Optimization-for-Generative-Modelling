import torch
import torch.nn as nn

HP_CONFIG = {
    # Core settings
    "MODEL_NAME": "PR",  # "Static_GM_GAN", "Vanilla", "PR"
    "latent_dim": 100,
    "image_dim": 784,  # for MNIST
    "K": 15,           # number of Gaussians

    # --- Mixture parameters (common values for initialization) ---
    "sigma": 0.2,
    "c": 0.3,

    # Loss configs
    "which_loss": 'PR',
    "which_div": 'Chi2', # Divergence type: 'chi2', 'KL', or 'rKL'
    "lambda": 0.05,

    # Training parameters
    "epochs": 200,
    "lr": 0.00001,
    "batch_size": 64,
    "gpus": -1, # -1 means use all available GPUs
    "logs": True,
    "num_samples_evaluate": 10000, # for evaluate_model
}

# Note: mus_static, mus, A, and weights will be initialized
# within the train function using values from HP_CONFIG to avoid global variable issues.
def get_hyperparameters():
    return HP_CONFIG