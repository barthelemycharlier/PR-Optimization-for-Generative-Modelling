import torch
import os
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def train(args):

    # Accept either an argparse.Namespace or a dict-like `args`.
    # Normalize to a dict `cfg` so the rest of the code can remain similar.
    if isinstance(args, dict):
        cfg = args
    else:
        try:
            cfg = vars(args)
        except TypeError:
            raise TypeError("train expects a dict or argparse.Namespace-like 'args'")

    # Extract hyperparameters from normalized config
    MODEL_NAME = cfg["MODEL_NAME"]
    latent_dim = cfg["latent_dim"]
    image_dim = cfg["image_dim"]
    K = cfg["K"]
    sigma = cfg["sigma"]
    c = cfg["c"]
    which_loss = cfg["which_loss"]
    which_div = cfg["which_div"]
    _lambda = cfg["lambda"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    gpus = cfg["gpus"]
    logs = cfg["logs"]
    num_samples = cfg["num_samples_evaluate"]

    to_download=True
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        if gpus == -1:
            gpus = torch.cuda.device_count()
            print(f"Using {gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")

    # Initialize mixture parameters *within* train function
    # Fixed centers for Static GM-GAN
    mus_static = torch.empty(K, latent_dim).uniform_(-c, c)

    # Learnable centers & linear transforms for Dynamic GM-GAN
    mus = nn.Parameter(torch.randn(K, latent_dim))
    A = nn.Parameter(torch.randn(K, latent_dim, latent_dim))

    # Mixture weights (uniform)
    weights = torch.ones(K) / K

    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True
        print("No dataset detected, to_download set to True")
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')

    print('Model loading...')
    mnist_dim = image_dim # Use image_dim from config
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    ## load new weights

    G.load_state_dict(torch.load("/content/drive/MyDrive/IASD/DSlab/help/checkpoints_static_c0.2_s1_K15_e200/G.pth"))
    D.load_state_dict(torch.load("/content/drive/MyDrive/IASD/DSlab/help/checkpoints_static_c0.2_s1_K15_e200/D.pth"))

    print("pretrained weights loaded from drive")


    GD = G_D(G, D)


    if gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        GD = nn.DataParallel(GD)

    print('Model loaded.')

    pr_config = {
        'which_loss': which_loss,
        'which_div': which_div,
        'lambda': _lambda,
        'latent_dim': latent_dim,
    }


    # # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # if MODEL_NAME == "PR":
    #   gen_loss_fn, discriminator_loss  = load_loss(pr_config)

    G_optimizer = optim.Adam(G.parameters(), lr=1e-6)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)


    perpare_curve = {
        'batch_size': batch_size,
        'num_pr_images': 10000,
        'dataset': 'mnist',
        'experiment_name': 'mnist_gan_test',
        'device': device,
        }

    # Allow for different batch sizes in G
  # G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = batch_size

    # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
    #                          device=device)
    def prepare_z_mixture(G_batch_size, dim_z, nclasses, mus, sigma=1.0,
                      device='cuda', fp16=False):
      # z from mixture of Gaussians
      z_ = Distribution(torch.empty(G_batch_size, dim_z, requires_grad=False))
      z_.init_distribution('mixture_normal', mus=mus, sigma=sigma)
      z_ = z_.to(device, torch.float16 if fp16 else torch.float32)
      if fp16:
          z_ = z_.half()

      # categorical labels (same as before)
      y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
      y_.init_distribution('categorical', num_categories=nclasses)
      y_ = y_.to(device, torch.int64)

      return z_, y_

    # z_, y_ = prepare_z_y(G_batch_size, latent_dim, 10 , fp16=False,
    #                          device=device)
    K = 15
    weights = torch.ones(K, device=device) / K
    c, sigma = 0.2, 1.0

    mus = torch.empty(K, latent_dim, device=device).uniform_(-c, c)


    z_, y_ = prepare_z_mixture(G_batch_size, latent_dim, 10, mus, sigma=1.0,
                           device=device)





    #### Prepare the training function

    train = GAN_training_function(G, D, GD, z_, y_,
                        config = cfg, D_optimizer = D_optimizer, G_optimizer = G_optimizer)

    print('Start training:')
    n_epoch = epochs
    running_G_loss = 0.0
    running_D_loss = 0.0
    num_batches = 0



    for epoch in range(1, n_epoch + 1):
        # D_optimizer.zero_grad()
        # G_optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)


            G.train()
            D.train()

            out = train(x,y)


            running_D_loss += out["D_loss_real"] + out["D_loss_fake"]
            running_G_loss += out["G_loss"]
            num_batches += 1
        avg_D_loss = running_D_loss / num_batches
        avg_G_loss = running_G_loss / num_batches
        print(f"[Epoch {epoch}] Avg D loss: {avg_D_loss:.4f}, Avg G loss: {avg_G_loss:.4f}")
        if epoch % 10 == 0 :
            if logs:
                results,_ = evaluate_model(
                    G,
                    device=device,
                    latent_dim=latent_dim,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    epoch=epoch,
                    version=MODEL_NAME,
                    lr=lr,
                    epochs=epochs,
                    avg_D_loss=avg_D_loss,
                    avg_G_loss=avg_G_loss,
                )

    print('Training done.')
    return G, D