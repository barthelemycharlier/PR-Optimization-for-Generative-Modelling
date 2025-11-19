
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

        # return torch.sigmoid(self.fc4(x))





class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        # return torch.sigmoid(self.fc4(x))
        return self.fc4(x)



class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(
        self,
        z,
        gy=None,          
        x=None,
        dy=None,         
        train_G=False,
        return_G_z=False,
        split_D=False
    ):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            G_z = self.G(z)


        # split_D: run D separately on fake and real
        if split_D:
            D_fake = self.D(G_z)
            if x is not None:
                D_real = self.D(x)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake

        # Non-split case: concatenate fake and real along the batch dimension
        if x is not None:
            D_input = torch.cat([G_z, x], dim=0)
        else:
            D_input = G_z

        D_out = self.D(D_input)

        if x is not None:
            # Split back into fake / real
            D_fake, D_real = torch.split(D_out, [G_z.shape[0], x.shape[0]], dim=0)
            if return_G_z:
                return D_fake, D_real, G_z
            else:
                return D_fake, D_real
        else:
            if return_G_z:
                return D_out, G_z
            else:
                return D_out


