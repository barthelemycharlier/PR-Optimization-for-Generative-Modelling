import numpy as np
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import torch.nn as nn
import argparse

from model import Generator

def create_inception_extractor(device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return model, tfm


def get_real_features(dataloader, model, tfm, device, max_images=None):
    feats = []
    total = 0

    with torch.no_grad():
        for x, _ in dataloader:
            if max_images and total >= max_images:
                break
            x = x.to(device)

            # x: (B,1,28,28) -> convert to 3 channels
            x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(299, 299))
            x = tfm(x)

            z = model(x).cpu().numpy()
            feats.append(z)

            total += x.size(0)

    return np.concatenate(feats, axis=0)


def get_fake_features(G, model, tfm, device, latent_dim, num_samples=5000, batch_size=64,version ="static"):
    feats = []
    G.eval()
    K = 15
    weights = torch.ones(K) / K  # uniform weights in the static case for the mixture
    c = 0.2
    sigma = 1
    batch_size = 64
    mus = torch.empty(K, latent_dim).uniform_(-c, c).to(device)
    A   = nn.Parameter(torch.randn(K, latent_dim, device=device))



    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            if version == "static":
              components = torch.multinomial(weights, batch_size, replacement=True) # sample from the gaussians with prob corresponding to weights
              z = mus[components] + sigma * torch.randn(batch_size, latent_dim, device=device)
            else :

              z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z)                            # (B, 784)
            fake = fake.view(-1, 1, 28, 28)        # reshape
            fake = fake.repeat(1, 3, 1, 1)
            fake = F.interpolate(fake, size=(299, 299))
            fake = tfm(fake)

            z = model(fake).cpu().numpy()
            feats.append(z)

    return np.concatenate(feats, axis=0)



def get_real_tensor(dataloader, num_samples):
    imgs = []
    for x, _ in dataloader:
        imgs.append(x)
        if sum(t.shape[0] for t in imgs) >= num_samples:
            break
    return torch.cat(imgs, dim=0)[:num_samples]


# def get_fake_tensor(G, latent_dim, num_samples, batch_size, device):
#     imgs = []
#     with torch.no_grad():
#         for _ in range(num_samples // batch_size):
#             z = torch.randn(batch_size, latent_dim, device=device)
#             fake = G(z).view(batch_size, 1, 28, 28)
#             imgs.append(fake)
#     return torch.cat(imgs, dim=0)[:num_samples]

def get_fake_tensor(G, latent_dim, num_samples, batch_size, device):
    imgs = []

    K = 15
    c = 0.2
    sigma = 1.0

    mus = torch.empty(K, latent_dim, device=device).uniform_(-c, c)
    weights = torch.ones(K, device=device) / K

    with torch.no_grad():
        for _ in range(num_samples // batch_size):

            # sample mixture components
            idx = torch.multinomial(weights, batch_size, replacement=True)
            base = mus[idx]

            # noise
            z = base + sigma * torch.randn_like(base)

            # generate
            fake = G(z).view(batch_size, 1, 28, 28)
            imgs.append(fake)

    return torch.cat(imgs, dim=0)[:num_samples]





import sys
sys.path.append('/content/drive/MyDrive/IASD/DSlab/help/')


import importlib
importlib.invalidate_caches()
import prd_score as prd

import precision_recall_kyn_utils

def compute_prd_from_model(G, dataloader, latent_dim, device="cuda", num_samples=5000):
    model, tfm = create_inception_extractor(device)

    real_feats = get_real_features(
        dataloader=dataloader,
        model=model,
        tfm=tfm,
        device=device,
        max_images=num_samples
    )

    fake_feats = get_fake_features(
        G=G,
        model=model,
        tfm=tfm,
        device=device,
        latent_dim=latent_dim,
        num_samples=num_samples,
        batch_size=64
    )

    precision, recall = prd.compute_prd_from_embedding(
        eval_data=fake_feats,
        ref_data=real_feats,
        num_clusters=20,
        num_angles=1001,
        num_runs=5,
        enforce_balance = False
    )
    real_imgs = get_real_tensor(dataloader, num_samples=num_samples).cuda()

    real_imgs = real_imgs.repeat(1, 3, 1, 1)

    fake_imgs = get_fake_tensor(G, latent_dim, num_samples, batch_size=64, device="cuda")
    fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

    ipr = precision_recall_kyn_utils.IPR(batch_size=64, k=3, num_samples=num_samples)
    ipr.compute_manifold_ref(real_imgs)
    precision_kyn, recall_kyn, density, coverage = ipr.precision_and_recall(fake_imgs)



    return precision, recall , precision_kyn, recall_kyn


def plot_prd_curve(precision, recall, label="Model", filename="prd_curve.png"):
    prd.plot([(precision, recall)], labels=[label], out_path=None)
    # print("Kynkäänniemi precision:", precision_kyn)
    # print("Kynkäänniemi recall   :", recall_kyn)


def main(args):
    # prepare device
    if isinstance(args.device, str):
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = args.device

    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    data_path = args.data_path

    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

    dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    latent_dim = args.latent_dim

    # create generator and optionally load weights
    G = Generator(g_output_dim=784).to(device)
    if args.g_path:
        state = torch.load(args.g_path, map_location=device)
        # support both state_dict and full model
        if isinstance(state, dict) and not any(k.startswith('__') for k in state.keys()):
            try:
                G.load_state_dict(state)
                print(f"Loaded Generator state_dict from {args.g_path}")
            except Exception:
                # maybe file contains {'model': state_dict} or similar
                if 'model' in state:
                    G.load_state_dict(state['model'])
                    print(f"Loaded Generator from nested 'model' in {args.g_path}")
                else:
                    print(f"Warning: could not directly load state dict from {args.g_path}")
        else:
            try:
                G = state.to(device)
                print(f"Loaded full Generator object from {args.g_path}")
            except Exception:
                print(f"Warning: unknown format for {args.g_path}; proceeding with random-initialized Generator")

    precision, recall, precision_kyn, recall_kyn = compute_prd_from_model(
        G,
        dataloader=dataloader,
        latent_dim=latent_dim,
        device=device,
        num_samples=args.num_samples,
    )

    print("Kynkäänniemi precision:", precision_kyn)
    print("Kynkäänniemi recall   :", recall_kyn)
    plot_prd_curve(precision, recall, label=f"lambda = 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PR / PRD for a Generator model")
    parser.add_argument("--g-path", type=str, default=None, help="Path to saved Generator state_dict or model")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device name or 'auto'")
    args = parser.parse_args()
    main(args)