"""Define the models for the Causal Variational Autoencoder (CVAE)"""

import torch
from torch.distributions import bernoulli
from torch.distributions import normal
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class p_x_z(nn.Module):
    """
    Class for the p(x|z) network in the CVAE model.
    """

    def __init__(
        self,
        dim_in=20,
        nh=3,
        dim_h=20,
        dim_out_bin=19,
        dim_out_con=6,
    ):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh - 2)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(0.1)

    def forward(self, z_input):
        z = F.relu(self.input(z_input))
        for i in range(self.nh - 2):
            z = F.relu(self.hidden[i](z))
            z = self.dropout(z)  # Shape: [batch_size, dim_h]
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)  # Shape: [batch_size, dim_out_bin]
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)  # Shape: [batch_size, dim_out_con]

        if (z != z).all():
            raise ValueError("p(x|z) forward contains NaN")

        return (
            x_bin,
            x_con,
        )


class p_t_z(nn.Module):
    """
    Class for the p(t|z) network in the CVAE model.
    This network is used to estimate the distribution of t given z.
    """

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.nh):
            x = F.relu(self.hidden[i](x))
            x = self.dropout(x)
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)  # Shape: [batch_size, dim_out]
        return out


class p_y_zt(nn.Module):
    """
    Class for the p(y|z,t) network in the CVAE model.
    This network is used to estimate the distribution of y given z and t.
    """

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR

        self.input_t0 = nn.Linear(dim_in, dim_h)
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, z, t):
        # t is a binary value of shape [batch_size, 1]
        # Separated forwards for different t values, TAR

        x_t0 = F.relu(self.input_t0(z))
        x_t0 = self.dropout(x_t0)
        for i in range(self.nh):
            x_t0 = F.relu(self.hidden_t0[i](x_t0))
            x_t0 = self.dropout(x_t0)
        mu_t0 = F.relu(self.mu_t0(x_t0))

        x_t1 = F.relu(self.input_t1(z))
        x_t1 = self.dropout(x_t1)
        for i in range(self.nh):
            x_t1 = F.relu(self.hidden_t1[i](x_t1))
            x_t1 = self.dropout(x_t1)
        mu_t1 = F.relu(self.mu_t1(x_t1))
        # set mu according to t value
        y = normal.Normal(
            (1 - t) * mu_t0 + t * mu_t1, 1
        )  # Output shape is [batch_size, dim_out]

        return y


# Inference model / Encoder
class q_t_x(nn.Module):
    """
    Class for the q(t|x) network in the CVAE model. This network is used to estimate the distribution of t given x.
    """

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.nh):
            x = F.relu(self.hidden[i](x))
            x = self.dropout(x)
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)  # Shape: [batch_size, dim_out]

        return out


class q_y_xt(nn.Module):
    """
    Class for the q(y|x,t) network in the CVAE model.
    This network is used to estimate the distribution of y given x and t.
    """

    def __init__(self, dim_in=25, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.relu(self.input(x))
        for i in range(self.nh):
            x = F.relu(self.hidden[i](x))
            x = self.dropout(x)
        # only output weights separated
        mu_t0 = self.mu_t0(x)  # Output shape is [batch_size, dim_out]
        mu_t1 = self.mu_t1(x)  # Output shape is [batch_size, dim_out]
        # set mu according to t, sigma set to 1
        y = normal.Normal(
            (1 - t.unsqueeze(1)) * mu_t0 + t.unsqueeze(1) * mu_t1, 1
        )  # Output shape is [batch_size, dim_out]
        return y


class q_z_tyx(nn.Module):
    """
    Class for the q(z|x,y,t) network in the CVAE model.
    This network is used to estimate the distribution of z given x, y and t.
    """

    def __init__(self, dim_in=25 + 1, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(0.1)

    def forward(self, xy, t):
        # Shared layers with separated output layers
        x = F.relu(self.input(xy))
        for i in range(self.nh):
            x = F.relu(self.hidden[i](x))
            x = self.dropout(x)

        mu_t0 = self.mu_t0(x)  # Output shape is [batch_size, dim_out]
        mu_t1 = self.mu_t1(x)  # Output shape is [batch_size, dim_out]
        sigma_t0 = self.softplus(
            self.sigma_t0(x)
        )  # Output shape is [batch_size, dim_out]
        sigma_t1 = self.softplus(
            self.sigma_t1(x)
        )  # Output shape is [batch_size, dim_out]

        # Set mu and sigma according to t
        z = normal.Normal(
            (1 - t.unsqueeze(1)) * mu_t0 + t.unsqueeze(1) * mu_t1,
            (1 - t.unsqueeze(1)) * sigma_t0 + t.unsqueeze(1) * sigma_t1,
        )  # Output shape is [batch_size, dim_out]
        return z
