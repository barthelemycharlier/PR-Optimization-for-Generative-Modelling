# PR-Optimization-for-Generative-Modelling
Done during the IASD master at PSL

## Precision-Recall Divergence optimization

The paper (Vérine et al., 2023) presents a novel approach to fine-tuning the precision–recall trade-off of a pre-trained GAN model. We briefly summarize the theory and switch the discriminator notation from $D$ to $T$ to avoid conflicts with divergence notation. In the original GAN paper (Goodfellow et al., 2014) an f-divergence $\mathcal{D}_f$ is used to measure the discrepancy between the target distribution $P$ and the model distribution $\widehat{P}_G$.

As a first (naive) approach one could try to minimize the dual variational form of the PR divergence $D_{\lambda\text{-PR}}$:

$$
\min_{G}\max_{T} \mathcal{D}^{\mathrm{dual}}_{f_\lambda, T}(P \;\|\; \widehat{P})
=
\min_{G}\max_{T} \mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim \widehat{P}}\left[f^{*}_{\lambda}(T(x))\right].
$$

However this formulation is prone to vanishing gradients. The paper therefore proposes to train the discriminator by maximizing the dual of an auxiliary divergence $D_g$ (with $g$ chosen appropriately):

$$
\max_{T} D^{\mathrm{dual}}_{g,T}(P \;\|\; \widehat{P}_G)
=
\mathbb{E}_{x\sim P}[T(x)] - \mathbb{E}_{x\sim \widehat{P}_G}\left[g^{*}(T(x))\right].
$$

The generator update is then approximated by optimizing the following objective (with $z\sim Q$ the latent noise):

$$
\min_{G} \; \mathbb{E}_{z\sim Q}\left[f_{\lambda}\left(\nabla g^{*}\big(T(G(z))\big)\right)\right].
$$

In practice this leads to the following training loop:

1. Sample real examples $x^{\mathrm{real}}_1, \dots, x^{\mathrm{real}}_N \sim P$ and fake examples $x^{\mathrm{fake}}_1, \dots, x^{\mathrm{fake}}_N \sim \widehat{P}_G$.
2. Update the parameters of $T$ by ascending the gradient

$$
\nabla \mathcal{L}_T = \frac{1}{N} \nabla \left\{ \sum_{i=1}^N T(x^{\mathrm{real}}_i) - \sum_{i=1}^N g^{*}(T(x^{\mathrm{fake}}_i)) \right\}.
$$

3. Update the parameters of $G$ by descending the gradient

$$
\nabla \mathcal{L}_G = \frac{1}{N} \nabla \left\{ \sum_{i=1}^N f\left(\nabla g^{*}(T(x^{\mathrm{fake}}_i))\right) \right\}.
$$

Because we use a GM-GAN as the pre-trained model, sampling from the latent distribution must respect the GM structure — this motivated the choice of a static GM-GAN where the mixture components are fixed and sampling is straightforward.

## PR Optimization

The PR optimization was performed using the auxiliary divergence $g = f\chi_2$ on a pre-trained "Static GM-GAN" (parameters: $c=0.2$, $s=1$, $K=15$). The models were trained with Adam for 200 epochs with learning rates $\mathrm{lr}=1\times 10^{-5}$ for the discriminator and $\mathrm{lr}=1\times 10^{-6}$ for the generator.

### Results

| Model | Precision | Recall |
|---|---:|---:|
| Static (baseline) | 0.26 | 0.50 |
| $\\lambda=0.05$ | 0.24 | 0.59 |
| $\\lambda=1$ | 0.22 | 0.55 |
| $\\lambda=10$ | 0.23 | 0.51 |
| $\\lambda=20$ | 0.19 | 0.51 |

![PRD curves](pr_pr.png)

We observe that the optimization improves recall for small $\\lambda$ (notably $\\lambda=0.05$) but increasing the trade-off parameter does not consistently improve precision. One possible explanation is that the base pre-trained model has a specific precision/recall bias; it may be easier to further increase recall of a high-recall model than to substantially increase precision.

Overall the approach appears promising and could yield better results when applied to a stronger base model. Implementation remains challenging due to limited documentation; we adapted the RejectBigGan implementation from the paper and modified it to work with a GM-GAN. PR-curve computation follows the method of Sajjadi et al. (2018) and precision/recall metrics use the method of Kynkäänniemi et al. (2019) with $k=3$, both computed with 10k samples.

