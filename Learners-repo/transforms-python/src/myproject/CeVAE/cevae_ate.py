# from importlib import reload
import logging
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import torch
import torch.distributions
import random
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
# from pyro.contrib.cevae import CEVAE
import pyro
from myproject.CeVAE.pyro_cevae import CEVAE
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)



def create_ate_df(roc, ate, combination, model, parameters):
    things = {}
    things["ROC"] = roc
    things["ATE"] = ate
    things["num_layers"] = parameters["num_layers"]
    things["hidden_dim"] = parameters["hidden_dim"]
    things["learning_rate"] = parameters["learning_rate"]
    things["num_epochs"] = parameters["num_epochs"]
    things["num_samples"] = parameters["num_samples"]
    things["latent_dim"] = parameters["latent_dim"]
    things["batch_size"] = parameters["batch_size"]
    things["drug_0"] = combination[0]
    things["drug_1"] = combination[1]
    things["model"] = model
    return pd.DataFrame(things, index=[0])

def batched_ite(cevae, X, batch_size):
    y0_all, y1_all = [], []

    for i in range(0, X.size(0), batch_size):
        Xb = X[i:i+batch_size]

        ites_b, y0_b, y1_b = cevae.ite(Xb)

        y0_all.append(y0_b.mean(dim=0).flatten())
        y1_all.append(y1_b.mean(dim=0).flatten())

        # VERY important
        del ites_b, y0_b, y1_b
        torch.cuda.empty_cache()

    y0 = torch.cat(y0_all)
    y1 = torch.cat(y1_all)
    return y0, y1

def cevae_func_foldless(param,
                        X_train_val_without_treatment,
                        y_train_val,
                        treatment_series,
                        combination,
                        results_df):
    """
    Train CEVAE once (foldless) on the provided X_train_val_without_treatment, treatment_series and y_train_val.
    Filters the input features to the provided 'features' list (if given).
    Computes individual potential outcomes y0,y1, ITEs and the ATE, and stores ROC & ATE in results_df.

    Args:
        param: dict of hyperparameters
        X_train_val_without_treatment: pandas.DataFrame (5 best covariates)
        y_train_val: pandas.Series or array-like (outcome)
        treatment_series: pandas.Series or array-like (0/1 treatment)
        combination: pair identifying drug_0 and drug_1 (for bookkeeping)
        results_df: pd.DataFrame to append results to
        features: list of column names to *keep* from X_train_val_without_treatment (optional but recommended)
    Returns:
        updated results_df (pd.DataFrame)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Make sure CPU fallback works
    if not torch.cuda.is_available():
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Convert to tensors
    X_tensor = torch.tensor(X_train_val_without_treatment.to_numpy(), dtype=torch.float32).to(device)
    t_tensor = torch.tensor(treatment_series.to_numpy(), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_val.to_numpy(), dtype=torch.float32).to(device)

    # Instantiate CEVAE with correct input dim
    cevae = CEVAE(
        feature_dim=X_tensor.size(1),
        num_layers=param["num_layers"],
        hidden_dim=param["hidden_dim"],
        latent_dim=param["latent_dim"],
        num_samples=param["num_samples"]
    ).to(device)

    # Fit once on the whole provided dataset
    cevae.fit(
        X_tensor,
        t_tensor,
        y_tensor,
        num_epochs=param["num_epochs"],
        learning_rate=param["learning_rate"],
        batch_size=param["batch_size"]
    )

    # Estimate potential outcomes (y0, y1) and ITEs on the *same* dataset (or you can pass a separate test set)
    # The model's .ite(X) should return (ites, y0, y1) or similar; match previous shape-handling.
    # We'll call it once and average across the sample dimension (same approach as your CV code).
    y0, y1 = batched_ite(cevae, X_tensor, batch_size=512)

    # # y0_samples and y1_samples expected shape: (num_samples, n_obs) or similar.
    # # average across sample dimension to get point estimates per observation.
    # # If dims differ, adjust accordingly.
    # y0 = y0_samples.mean(dim=0).flatten()   # shape: (n_obs,)
    # y1 = y1_samples.mean(dim=0).flatten()

    # Individual treatment effect
    ite_estimates = (y1 - y0).detach().cpu().numpy()  # numpy array (n_obs,)

    # ATE: mean ITE
    ate = float(np.mean(ite_estimates))

    # For ROC: build predicted outcome under observed treatment and compute ROC AUC
    # y_pred_obs = t * y1 + (1-t) * y0
    y_pred_obs = torch.where(t_tensor > 0, y1, y0).detach().cpu().numpy()
    y_true = y_tensor.detach().cpu().numpy()

    # Some corner cases: ROC requires both classes present
    try:
        roc = float(roc_auc_score(y_true, y_pred_obs))
    except Exception as e:
        logging.warning(f"ROC compute failed: {e}. Setting ROC = np.nan")
        roc = np.nan

    # Append to results_df
    out_df = create_ate_df(roc, ate, combination, "CEVAE", param)
    results_df = pd.concat([results_df, out_df], ignore_index=True)

    # Clean up
    del cevae
    torch.cuda.empty_cache()

    return results_df
