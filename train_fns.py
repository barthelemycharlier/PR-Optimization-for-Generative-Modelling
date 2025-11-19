import torch
from losses import load_loss, rate


def GAN_training_function(
    G, D, GD, z_, y_, config, D_optimizer, G_optimizer
):
    generator_loss, discriminator_loss = load_loss(config)
    discriminator_rate = rate(config)

    num_D_steps =  1
    num_G_steps =  1
    num_D_accum = 1
    num_G_accum = 1

    def train(x, y, train_G=True):
        # Split real data into chunks matching batch_size
        x = torch.split(x, config["batch_size"])
        y = torch.split(y, config["batch_size"])
        counter = 0


        for _ in range(num_D_steps):
            D_optimizer.zero_grad()
            for _ in range(num_D_accum):
                z_.sample_()
                y_.sample_()
                D_fake, D_real = GD(
                    z_[:config["batch_size"]],
                    None,
                    x[counter],
                    None,
                    train_G=False,
                    split_D=True
                )
                D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / num_D_accum
                D_loss.backward()
                counter += 1
            D_optimizer.step()

        G_loss = torch.tensor(0.0, device=x[0].device)
        if train_G:
            G_optimizer.zero_grad()
            counter = 0
            for _ in range(num_G_accum):
                batch = x[counter].size(0)
                z_.sample_()
                y_.sample_()
                z_input = z_[:batch]
                D_fake, D_real = GD(
                    z_input,
                    None,
                    x[counter],
                    None,
                    train_G=train_G,
                    split_D=config.get("split_D", False)
                )

                g_loss = generator_loss(D_real, D_fake)
                g_loss = g_loss / num_G_accum
                g_loss.backward()
                G_loss += g_loss
                counter += 1
            G_optimizer.step()

        out = {
            "G_loss": float(G_loss.item()),
            "D_loss_real": float(D_loss_real.item()),
            "D_loss_fake": float(D_loss_fake.item()),
            "Acc_real": float(
                torch.sum(discriminator_rate(D_real)) / D_real.size(0) * 100
            ),
            "Acc_fake": float(
                100 - torch.sum(discriminator_rate(D_fake)) / D_fake.size(0) * 100
            ),
        }
        return out

    return train
