import torch

class A_VAE_Loss(torch.nn.Module):
    def __init__(self):
        super(A_VAE_Loss, self).__init__()
        self.mseloss = torch.nn.MSELoss()

    def forward(self, mu, logvar, recon_x, x):

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BCE = self.mseloss(recon_x, x)
        elbo = KLD + BCE

        return elbo, KLD, BCE


class T_VAE_Loss(torch.nn.Module):

    def __init__(self):
        super(T_VAE_Loss, self).__init__()
        self.nlloss = torch.nn.NLLLoss()

    def KL_loss(self, mu, log_var, z):
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(-1)
        kl = kl.mean()

        return kl

    def reconstruction_loss(self, x_hat_param, x):
        x = x.view(-1).contiguous()
        x_hat_param = x_hat_param.view(-1, x_hat_param.size(2))

        recon = self.nlloss(x_hat_param, x)

        return recon

    def forward(self, mu, log_var, z, x_hat_param, x):
        kl_loss = self.KL_loss(mu, log_var, z)
        recon_loss = self.reconstruction_loss(x_hat_param, x)

        elbo = kl_loss + recon_loss

        return elbo, kl_loss, recon_loss