"""Function for training and validation of the model."""

import torch
from torch.nn import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            # print(f'Gradient before clip: {params[0].grad}')
            utils.clip_grad_norm_(params, max_norm=1.0, norm_type=2)
            # print(f'Gradient after clip: {params[0].grad}')
            # Update step
            optimizer.step()

        loss += objective.item()

    return loss
