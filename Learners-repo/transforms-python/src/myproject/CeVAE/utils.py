import torch
import torch.optim as optim
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
