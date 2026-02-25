# from importlib import reload
import logging

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils
import torch.distributions
from torch.distributions import normal
from torch.distributions import bernoulli
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
import pyro
import pyro.distributions as dist
# from pyro.contrib.cevae import CEVAE
from myproject.CeVAE.pyro_cevae import CEVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import dataset_script
# import utils
# import networks
# import evaluation
# import train_script

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(object):
    def __init__(self, main_df, combination):
        # Initialize instance variables
        self.main_df = main_df
        self.combination = combination
        self.results_df = pd.DataFrame()
        self.binfeats = [
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            # 32,
        ]  # Binary features
        self.confeats = [1]  # Continous features

    def get_train_valid_test(self):
        df = self.main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(self.combination))].reset_index(
            drop=True
        )
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == self.combination[0] else 1
        )

        # Set mu0 and mu1
        df["mu0"] = df[df["treatment"] == 0]["severity_final"].mean()
        df["mu1"] = df[df["treatment"] == 1]["severity_final"].mean()

        mu0 = df["mu0"].reset_index(drop=True)
        mu1 = df["mu1"].reset_index(drop=True)

        X = df.drop(
            [
                "person_id",
                "severity_final",
                "ingredient_concept_id",
                "treatment",
                "mu0",
                "mu1",
            ],
            axis=1,
        )
        X["age_at_covid"] = (X["age_at_covid"] - X["age_at_covid"].min()) / (
            X["age_at_covid"].max() - X["age_at_covid"].min()
        )
        X = X.reset_index(drop=True)
        y = df["severity_final"]
        T = df["treatment"]

        np.random.seed(42)
        idxtrain, ite = model_selection.train_test_split(
            np.arange(X.shape[0]), test_size=0.1, random_state=1
        )
        itr, iva = model_selection.train_test_split(
            idxtrain, test_size=0.3, random_state=1
        )
        train = (X.iloc[itr], T.iloc[itr], y.iloc[itr]), (mu0.iloc[itr], mu1.iloc[itr])
        valid = (X.iloc[iva], T.iloc[iva], y.iloc[iva]), (mu0.iloc[iva], mu1.iloc[iva])
        test = (X.iloc[ite], T.iloc[ite], y.iloc[ite]), (mu0.iloc[ite], mu1.iloc[ite])
        return train, valid, test, self.binfeats, self.confeats


class Evaluator(object):
    def __init__(self, y, t, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def estimate_ate(self, ypred1, ypred0):
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        ate = np.mean(ite1 - ite0)
        return ate

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def roc(self, y1, y0):
        ypred = (1 - self.t) * y0 + self.t * y1
        roc = metrics.roc_auc_score(self.y, ypred)
        return roc

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        return ite, ate


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


def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, L=1):
    """
    Function to get y0 (control) and y1 (treated) from the model

    Args:
        p_y_zt_dist (dist): Distribution of y given z and t of shape (batch_size, 1)
        q_y_xt_dist (dist): Distribution of y given x and t of shape (batch_size, 1)
        q_z_tyx_dist (dist): Distribution of z given x, y and t
        x_train (tensor): Input data of shape (batch_size, dim_x)
        t_train (tensor): Treatment data of shape (batch_size, 1)

    Returns:
        y0 (array): Mean of the distribution of y given z and t=0 (control)
        y1 (array): Mean of the distribution of y given z and t=1 (treated)
    """
    y_infer = q_y_xt_dist(x_train.float(), t_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_tyx_dist(
        xy=xy, t=t_train.float()
    )  # z_infer is the distribution of z given x, y and t
    z_infer_mean = z_infer.mean
    # Manually input zeros and ones
    y0 = p_y_zt_dist(
        z_infer_mean, torch.zeros(z_infer_mean.shape).to(device)
    ).loc.mean(  # .cuda()
        dim=1
    )  # y0 is the mean of the distribution of y given z and t=0 (control)
    y1 = p_y_zt_dist(
        z_infer_mean, torch.ones(z_infer_mean.shape).to(device)
    ).loc.mean(  # .cuda()
        dim=1
    )  # y1 is the mean of the distribution of y given z and t=1 (treated)

    return y0.cpu().detach().numpy(), y1.cpu().detach().numpy()


def init_qz(qz, y, t, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step

    Args:
        qz (nn.Module): qz network
        y (array): Outcome data of shape (batch_size,)
        t (array): Treatment data of shape (batch_size,)
        x (array): Input data of shape (batch_size, len(binfeats) + len(contfeats))

    Returns:
        qz (nn.Module): Initialized qz network
    """
    idx = list(range(x.shape[0]))  # list of indices
    np.random.shuffle(idx)  # shuffle indices

    optimizer = optim.Adamax(qz.parameters(), lr=1e-4)

    for i in range(50):
        batch = np.random.choice(idx, 1)  # random batch
        x_train, y_train, t_train = (
            torch.FloatTensor(x[batch]).to(device),  # convert x to tensor
            torch.FloatTensor(y[batch]).to(device),  # convert y to tensor
            torch.FloatTensor(t[batch]).to(device),  # convert t to tensor
        )
        xy = torch.cat(
            (x_train, y_train.unsqueeze(1)), 1
        )  # concatenate x and y. Shape: (batch_size, 2)

        # print(xy.shape, t_train.shape)

        z_infer = qz(xy=xy, t=t_train)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (
            -torch.log(z_infer.stddev)
            + 1 / 2 * (z_infer.variance + z_infer.mean**2 - 1)
        ).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError("KL(pz,qz) contains NaN during init")

    return qz


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def create_best_params_df(val_roc, test_roc, test_ate, best_params, combination, model):
    best_params["val_roc"] = val_roc
    best_params["test_roc"] = test_roc
    best_params["test_ate"] = test_ate
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


def train_val(
    loader,
    optimizer,
    params,
    binfeats,
    contfeats,
    q_z_tyx_dist,
    p_x_z_dist,
    p_t_z_dist,
    p_y_zt_dist,
    p_z_dist,
    q_t_x_dist,
    q_y_xt_dist,
    train=True,
):
    loss = 0.0
    for x_t, y_t, t_t in loader:
        x_t, y_t, t_t = (
            x_t.to(device),
            y_t.to(device),
            t_t.to(device),
        )
        xy = torch.cat((x_t, y_t.unsqueeze(1)), dim=1)
        z_infer = q_z_tyx_dist(xy=xy, t=t_t)  # Shape: [batch_size, z_dim]
        # use a single sample to approximate expectation in lowerbound
        z_infer_sample = z_infer.sample()  # Shape: [batch_size, z_dim]

        # RECONSTRUCTION LOSS
        # p(x|z)
        # x_bin is the distribution of binary features given z,
        # x_con is the distribution of continuous features given z
        x_bin, x_con = p_x_z_dist(
            z_infer_sample
        )  # Shape: [batch_size, len(binfeats)], [batch_size, len(contfeats)]

        l1 = x_bin.log_prob(x_t[:, : len(binfeats)]).sum(
            dim=1
        )  # Shape: [batch_size, 1]

        l2 = x_con.log_prob(x_t[:, -len(contfeats) :]).sum(
            dim=1
        )  # Shape: [batch_size, 1]

        t = p_t_z_dist(z_infer_sample)  # distribution of t given z
        l3 = t.log_prob(t_t).squeeze()  # Shape: [batch_size, 1]

        y = p_y_zt_dist(z_infer_sample, t_t)  # distribution of y given z and t
        l4 = y.log_prob(y_t).squeeze()  # Shape: [batch_size, 1]

        # REGULARIZATION LOSS
        # p(z) - q(z|x,t,y)
        # approximate KL
        l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(
            dim=1
        )  # Shape: [batch_size, 1]

        # AUXILIARY LOSS
        # q(t|x)
        t_infer = q_t_x_dist(x_t)  # t_infer is the distribution of t given x
        l6 = t_infer.log_prob(t_t).squeeze()  # Shape: [batch_size, 1]

        # q(y|x,t)
        y_infer = q_y_xt_dist(
            x_t, t_t
        )  # y_infer is the distribution of y given x and t
        l7 = y_infer.log_prob(y_t).squeeze()  # Shape: [batch_size, 1]

        # Total objective
        # inner sum to calculate loss per item, torch.mean over batch
        loss_mean = torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)  # Shape: [1]
        objective = -loss_mean

        if train:
            optimizer.zero_grad()
            # Calculate gradients
            objective.backward()
            # Gradient clipping
            utils.clip_grad_norm_(params, max_norm=1.0, norm_type=2)
            # Update step
            optimizer.step()

        loss += objective.item()

    return loss


def cevae_func(main_df, ingredient_pairs, param_grid, epochs):
    """
    Implement CEVAE method

    Args:
        main_df: patient records for a given treatment-control group
        ingredient_pairs: treatments
    """
    results_df = pd.DataFrame()
    for idx, combination in enumerate(ingredient_pairs):
        logger.info(
            f"-----------Drug pair: {combination}.{idx + 1}|{len(ingredient_pairs)} -----------"
        )
        print(
            f"-----------Drug pair: {combination}.{idx + 1}|{len(ingredient_pairs)} -----------"
        )
        dataset = CustomDataset(main_df, combination)

        train, valid, test, binfeats, contfeats = dataset.get_train_valid_test()

        # read out data
        (xtr, ttr, ytr), (_, _) = train
        (xva, tva, yva), (mu0va, mu1va) = valid
        (xte, tte, yte), (mu0te, mu1te) = test

        # reorder features with binary first and continuous after
        perm = binfeats + contfeats
        xtr, xva, xte = xtr.iloc[:, perm], xva.iloc[:, perm], xte.iloc[:, perm]
        xtr, xva, xte = xtr.to_numpy(), xva.to_numpy(), xte.to_numpy()
        ytr, yva, yte = ytr.to_numpy(), yva.to_numpy(), yte.to_numpy()
        ttr, tva, tte = ttr.to_numpy(), tva.to_numpy(), tte.to_numpy()

        # concatenate train and valid for training
        x_all_tr, t_all_tr, y_all_tr = (
            np.concatenate([xtr, xva], axis=0),
            np.concatenate([ttr, tva], axis=0),
            np.concatenate([ytr, yva], axis=0),
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        evaluator_test = Evaluator(yte, tte, mu0=mu0te, mu1=mu1te)

        best_roc = -np.Inf
        best_params = None

        for param in ParameterGrid(param_grid):
            logger.info(f"Training with params: {param}")
            print(f"Training with params: {param}")
            fold_scores = []

            # Training and cross-validation phase
            for train_idx, val_idx in skf.split(x_all_tr, t_all_tr):
                xtr_fold, xva_fold = x_all_tr[train_idx], x_all_tr[val_idx]
                ytr_fold, yva_fold = y_all_tr[train_idx], y_all_tr[val_idx]
                ttr_fold, tva_fold = t_all_tr[train_idx], t_all_tr[val_idx]

                # set evaluator objects
                # evaluator_train = Evaluator(
                #     ytr,
                #     ttr,
                #     mu0=np.concatenate([mu0tr, mu0va], axis=0),
                #     mu1=np.concatenate([mu1tr, mu1va], axis=0),
                # )
                evaluator_val = Evaluator(yva_fold, tva_fold, mu0=mu0va, mu1=mu1va)

                # Create dataset and dataloader
                train_dataset = TensorDataset(
                    torch.FloatTensor(xtr_fold),
                    torch.FloatTensor(ytr_fold),
                    torch.FloatTensor(ttr_fold),
                )
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

                validation_dataset = TensorDataset(
                    torch.FloatTensor(xva_fold),
                    torch.FloatTensor(yva_fold),
                    torch.FloatTensor(tva_fold),
                )
                validation_loader = DataLoader(
                    validation_dataset, batch_size=128, shuffle=False
                )

                # init networks (overwritten per replication)
                x_dim = len(binfeats) + len(contfeats)
                p_x_z_dist = p_x_z(
                    dim_in=param["z_dim"],
                    nh=3,
                    dim_h=param["h_dim"],
                    dim_out_bin=len(binfeats),
                    dim_out_con=len(contfeats),
                ).to(device)
                p_t_z_dist = p_t_z(
                    dim_in=param["z_dim"], nh=1, dim_h=param["h_dim"], dim_out=1
                ).to(device)
                p_y_zt_dist = p_y_zt(
                    dim_in=param["z_dim"], nh=1, dim_h=param["h_dim"], dim_out=1
                ).to(device)
                q_t_x_dist = q_t_x(
                    dim_in=x_dim, nh=1, dim_h=param["h_dim"], dim_out=1
                ).to(device)
                # t is not fed into network, therefore not increasing input size (y is fed).
                q_y_xt_dist = q_y_xt(
                    dim_in=x_dim, nh=1, dim_h=param["h_dim"], dim_out=1
                ).to(device)
                q_z_tyx_dist = q_z_tyx(
                    dim_in=len(binfeats) + 1 + len(contfeats),
                    nh=1,
                    dim_h=param["h_dim"],
                    dim_out=param["z_dim"],
                ).to(device)
                p_z_dist = normal.Normal(
                    torch.zeros(param["z_dim"]).to(device),
                    torch.ones(param["z_dim"]).to(device),
                )

                # Create optimizer
                params = (
                    list(p_x_z_dist.parameters())
                    + list(p_t_z_dist.parameters())
                    + list(p_y_zt_dist.parameters())
                    + list(q_t_x_dist.parameters())
                    + list(q_y_xt_dist.parameters())
                    + list(q_z_tyx_dist.parameters())
                )

                # Adamax is used, like original implementation
                # optimizer = optim.Adamax(params, lr=param["lr"], weight_decay=1e-5)

                # init q_z inference
                # q_z_tyx_dist = init_qz(q_z_tyx_dist, ytr, ttr, xtr).to(device)
                ytr_tensor = torch.from_numpy(ytr).to(torch.float32)
                # logging.log(ytr_tensor)
                # logging.log(ytr.size())
                ttr_tensor = torch.from_numpy(ttr).to(torch.float32)
                xtr_tensor = torch.from_numpy(xtr).to(torch.float32)
                cevae = CEVAE(feature_dim=xtr_tensor.size(1))
                losses = cevae.fit(xtr_tensor, ttr_tensor, ytr_tensor)
                
                # early_stopping = EarlyStopping(patience=5, min_delta=0.05)

                # for epoch in range(epochs):
                #     train_loss = 0.0

                #     # Training
                #     train_loss = train_val(
                #         train_loader,
                #         optimizer,
                #         params,
                #         binfeats,
                #         contfeats,
                #         q_z_tyx_dist,
                #         p_x_z_dist,
                #         p_t_z_dist,
                #         p_y_zt_dist,
                #         p_z_dist,
                #         q_t_x_dist,
                #         q_y_xt_dist,
                #         train=True,
                #     )

                #     # Validation
                #     with torch.no_grad():
                #         val_loss = train_val(
                #             validation_loader,
                #             optimizer,
                #             params,
                #             binfeats,
                #             contfeats,
                #             q_z_tyx_dist,
                #             p_x_z_dist,
                #             p_t_z_dist,
                #             p_y_zt_dist,
                #             p_z_dist,
                #             q_t_x_dist,
                #             q_y_xt_dist,
                #             train=False,
                #         )

                #     logger.info(
                #         f"Epoch: {epoch + 1}|{epochs}, train loss: {train_loss}, val loss: {val_loss}"
                #     )
                #     print(
                #         f"Epoch: {epoch + 1}|{epochs}, train loss: {train_loss}, val loss: {val_loss}"
                #     )

                #     # Check early stopping
                #     early_stopping(val_loss)
                #     if early_stopping.early_stop:
                #         print(f"Early stopping at epoch {epoch + 1}")
                #         logger.info(f"Early stopping at epoch {epoch + 1}")
                #         break

                # y0_val, y1_val = get_y0_y1(
                #     p_y_zt_dist,
                #     q_y_xt_dist,
                #     q_z_tyx_dist,
                #     torch.tensor(xva_fold).to(device),
                #     torch.tensor(tva_fold).to(device),
                # )

                val_losses = []
                val_y1s = []
                val_y0s = []
                for x_t, y_t, t_t in validation_loader:
                    val_loss, y1, y0 = cevae.ite(x_t)
                    val_losses.append(val_loss)
                    val_y1s.append(y1)
                    val_y0s.append(y0)
                
                # Calculate ROC for the val and test set
                roc_auc_val = evaluator_val.roc(val_y1s, val_y0s)
                fold_scores.append(roc_auc_val)

            mean_score = np.mean(fold_scores)
            if mean_score > best_roc:
                best_params = param
                best_roc = mean_score

        logger.info(f"Best params: {best_params}")
        logger.info(f"Best ROC AUC score: {best_roc}")
        print(f"Best params: {best_params}")
        print(f"Best ROC AUC score: {best_roc}")

        y0_test, y1_test = get_y0_y1(
            p_y_zt_dist,
            q_y_xt_dist,
            q_z_tyx_dist,
            torch.tensor(xte).to(device),
            torch.tensor(tte).to(device),
        )

        test_ate = evaluator_test.abs_ate(y1_test, y0_test)  # Test ATE
        roc_auc_test = evaluator_test.roc(y1_test, y0_test)  # Test ROC

        params_df = create_best_params_df(
            roc_auc_val, roc_auc_test, test_ate, best_params, combination, "CEVAE"
        )

        results_df = pd.concat([results_df, params_df], ignore_index=True)

    return results_df
