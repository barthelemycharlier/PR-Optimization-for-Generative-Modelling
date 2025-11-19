import torch
import torch.nn.functional as F
import torch.nn as nn

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss

def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss


# f-div Dual loss
class pr_loss_dis(torch.nn.Module):
  def __init__(self, config):
    super(pr_loss_dis, self).__init__()
    self.l = config['lambda']

  def forward(self, dis_fake, dis_real):
    # print(f'\nBefore {dis_fake.min():.2f}/{dis_fake.max():.2f}, {dis_real.min():.2f}/{dis_real.max():.2f}')

    dis_fake = torch.clamp(dis_fake, min=0)
    # dis_real = torch.clamp(dis_real, max=self.l )
    # print(f'After {dis_fake.mean():.2f} ({dis_fake.min():.2f}/{dis_fake.max():.2f}),  {dis_real.mean():.2f} ({dis_real.min():.2f}/{dis_real.max():.2f})')

    loss_real = -torch.mean(dis_real)
    loss_fake = torch.mean(dis_fake/self.l)
    return loss_real, loss_fake

class pr_loss_gen(torch.nn.Module):
  def __init__(self, config):
    super(pr_loss_gen, self).__init__()
    self.l = config['lambda']

  def forward(self, dis_fake):
    dis_fake = torch.clamp(dis_fake, max=self.l)

    loss_fake = -torch.mean(dis_fake/self.l)
    return loss_fake

def gan_loss_dis(dis_fake, dis_real):
  dis_fake = torch.sigmoid(dis_fake)
  dis_real = torch.sigmoid(dis_real)
  return -torch.log(1e-5+dis_real).mean(), -torch.log(1e-5+1-dis_fake).mean()


def gan_loss_gen(dis_fake):
  dis_fake = torch.sigmoid(dis_fake)
  return -torch.log(1e-5+dis_fake).mean()

def gan_loss_rej(dis_fake, a, K):
  dis_fake = torch.sigmoid(dis_fake)
  u = dis_fake/((1e-5+1-dis_fake)*(1e-5+K*a))
  return (K*a*(u*torch.log(1e-5+u)-(u+1)*torch.log(u+1))).mean()



def rkl_loss_dis(dis_fake, dis_real):
  # print(f'\nBefore {dis_fake.min():.2f}/{dis_fake.max():.2f}, {dis_real.min():.2f}/{dis_real.max():.2f}')

  dis_fake = -torch.exp(-dis_fake/10)
  dis_real = -torch.exp(-dis_real/10)
  dis_fake = torch.clamp(dis_fake, min=-5)
  # print(f'After {dis_fake.mean():.2f} ({dis_fake.min():.2f}/{dis_fake.max():.2f}),  {dis_real.mean():.2f} ({dis_real.min():.2f}/{dis_real.max():.2f})')

  loss_real = -torch.mean(dis_real)
  loss_fake = torch.mean(-1 - torch.log(-dis_fake +1e-3))
  return loss_real, loss_fake

def rkl_loss_gen(dis_fake):
  dis_fake = -torch.exp(-dis_fake/10)
  # dis_fake = torch.clamp(dis_fake,)
  loss_fake = -torch.mean(-1 - torch.log(-dis_fake +1e-3))
  return loss_fake



def kl_loss_dis(dis_fake, dis_real):
  dis_fake = torch.clamp(dis_fake, max=10)
  dis_real = torch.clamp(dis_fake, min=-100)
  loss_real = -torch.mean(dis_real)
  loss_fake = torch.mean(torch.exp(dis_fake-1))
  return loss_real, loss_fake

def kl_loss_gen(dis_fake):
  dis_fake = torch.clamp(dis_fake, max=10)
  loss_fake = -torch.mean(torch.exp(dis_fake-1))
  return loss_fake

def kl_loss_rej(dis_fake, a, K):
  dis_fake = torch.clamp(dis_fake, max=10)
  pph = torch.exp(dis_fake-1)
  ppt = pph/(K*a+1e-5)
  return (K*a*ppt*torch.log(pph+1e-5)).mean()

def loss_hinge_rej(dis_fake, a, K):
  loss = torch.mean(K*a*F.softplus(-dis_fake))

  return loss


def chi2_loss_dis(dis_fake, dis_real):
  dis_fake = torch.clamp(dis_fake, min=-2)
  dis_real = torch.clamp(dis_real, max=20)
  loss_real = -torch.mean(dis_real)
  loss_fake = torch.mean((dis_fake**2)/4+dis_fake)

  return loss_real, loss_fake

def chi2_loss_gen(dis_fake):
  dis_fake = torch.clamp(dis_fake, min=-2)
  loss_fake = -torch.mean((dis_fake**2)/4+dis_fake)
  return loss_fake


# PR primal loss
class PRLoss(nn.Module):
  def __init__(self, config):
    super(PRLoss, self).__init__()
    self.div = config['which_div']
    self.l = config['lambda']
    if self.div == "rKL":
      def rate(Dx):
        Dx = torch.clamp(Dx, max=0)
        return -1/(-1e-3+Dx)
    elif self.div == "KL":
      def rate(Dx):
        return torch.exp(Dx-1)
    elif self.div == "Chi2":
      def rate(Dx):
        Dx = torch.clamp(Dx, min=-2, max=5)
        return Dx/2+1
    self.rate = rate

  def alpha_train(self, pqr, pqf, lbd):
    a = torch.clamp(lbd*pqf, max=1).mean()
    return a

  def alpha(self, pqr, pqf, lbd):
    a = torch.clamp(lbd*pqf, max=1)*0.5 + torch.clamp(1/pqr, max=lbd)*0.5
    return a.mean()

  def beta_train(self, pqr, pqf, lbd):
    b = torch.clamp(1/(pqr*lbd), max=1).mean()
    return b


  def beta(self, pqr, pqf, lbd):
    b = torch.clamp(pqf, max=1/lbd)*0.5 + torch.clamp(1/(pqr*lbd), max=1)*0.5
    return b.mean()

  def forward(self, Dxr, Dxf):
    pqr = self.rate(Dxr)
    pqf = self.rate(Dxf)
    loss = - self.alpha(pqr, pqf, self.l)*0.5 - self.beta(pqr, pqf, self.l)*0.5
    return loss

def rate(config):
  if config['which_loss'] == 'vanilla':
    def rate_vanilla(Dx):
      return Dx>1
    return rate_vanilla
  else:
    if config['which_div'] == 'Chi2':
      def ratechi2(Dx):
        return  Dx>0
      return ratechi2
    elif config['which_div'] == 'KL':
      def ratekl(Dx):
        return  Dx>1
      return ratekl
    elif config['which_div'] == 'rKL':
      def ratekl(Dx):
        return  Dx>-1
      return ratekl
    elif config['which_div'] == 'pr':
      def ratepr(Dx):
        return  Dx>config['lambda']/2
      return ratepr
    elif config['which_div'] == "gan" or config['which_div'] == "hinge":
      def rategan(Dx):
        return Dx>0
      return rategan


def load_loss(config):
  # if config['which_loss'] == 'vanilla':
  #   return loss_hinge_gen, loss_hinge_dis

  # elif config['which_loss'] == 'div':
  #   if config['which_div'] == 'Chi2':
  #     return chi2_loss_gen, chi2_loss_dis
  #   elif config['which_div'] == 'KL':
  #     return kl_loss_gen, kl_loss_dis
  #   elif config['which_div'] == 'rKL':
  #     return rkl_loss_gen, rkl_loss_dis
  #   elif config['which_div'] == 'pr':
  #     return pr_loss_gen(config), pr_loss_dis(config)



  if config['which_loss'] == 'PR':
    if config['which_div'] == 'Chi2':
      loss_dis = chi2_loss_dis
    elif config['which_div'] == 'KL':
      loss_dis = kl_loss_dis
    elif config['which_div'] == 'rKL':
      loss_dis = rkl_loss_dis
    return PRLoss(config), loss_dis


  # elif config['which_loss'] == 'reject':
  #   if config['which_div'] == 'KL':
  #     return kl_loss_rej, kl_loss_dis
  #   elif config['which_div'] == 'gan':
  #     return gan_loss_rej, gan_loss_dis
  #   elif config['which_div'] == 'hinge':
  #     return loss_hinge_rej, gan_loss_dis



def pq_fun(config):
  if config['which_loss'] == "vanilla":
    def rate(Dx):
      return torch.exp(Dx)
    return rate
  if config['which_div'] == "rKL":
    def rate(Dx):
      Dx = torch.clamp(Dx, max=0)
      return -1/(-1e-3+Dx)
    return rate
  elif config['which_div'] == "KL":
    def rate(Dx):
      return torch.exp(Dx-1)
    return rate
  elif config['which_div'] == "Chi2":
    def rate(Dx):
      Dx = torch.clamp(Dx, min=-2, max=20)+1e-3
      return Dx/2+1
    return rate
  elif config['which_div'] == "gan" or config['which_div'] == "hinge":
    def rate(Dx):
      Dx = torch.sigmoid(Dx)
      return Dx/(1-Dx+1e-5)
    return rate
# # # Default to hinge loss
# # generator_loss = loss_hinge_gen
# # discriminator_loss = loss_hinge_dis
# generator_loss = kl_loss_gen
# discriminator_loss = kl_loss_dis
