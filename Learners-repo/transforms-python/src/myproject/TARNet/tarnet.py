# from pyspark.sql.functions import col, lit, when
# from pyspark.sql import SparkSession

# import palantir_models_serializers as pms

# Sklearn packages
from sklearn.utils import class_weight, resample
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)

# from transforms.api import transform, Input, Output, configure
# from multiprocessing import Pool
import logging
import copy

import torch
import torch.nn as nn
import torch.optim as optim

# import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# from itertools import combinations

logger = logging.getLogger()
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None  # Store the best model here

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss
        self.best_model = copy.deepcopy(model.state_dict())  # Deep copy the model state

    def load_best_weights(self, model):
        """Load the best model weights saved so far."""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)


class TARNet(nn.Module):
    def __init__(
        self,
        input_size,
        shared_representation_size,
        outcome_representation_size,
        dropout,
    ):
        super(TARNet, self).__init__()

        # Shared representation layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, shared_representation_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
            # You can add more layers here
        )

        # Separate heads for treated and control
        self.treated_head = nn.Sequential(
            nn.Linear(shared_representation_size, outcome_representation_size),
            nn.ReLU(),
            nn.Linear(outcome_representation_size, 1),  # Predicting outcome for treated
        )

        self.control_head = nn.Sequential(
            nn.Linear(shared_representation_size, outcome_representation_size),
            nn.ReLU(),
            nn.Linear(outcome_representation_size, 1),  # Predicting outcome for control
        )

    def forward(self, x, treatment):
        shared_rep = self.shared_layers(x)

        # Apply the appropriate head based on treatment
        treated_pred = self.treated_head(shared_rep)
        control_pred = self.control_head(shared_rep)

        # Combine predictions
        # This assumes treatment is a binary tensor of 0s and 1s
        preds = treatment * treated_pred.squeeze(-1) + (
            1 - treatment
        ) * control_pred.squeeze(-1)
        return preds, treated_pred, control_pred


class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size,
        hidden_size,
        criterion,
        optimizer_class,
        lr,
        batch_size,
        epochs,
        dropout,
        X_val=None,
        y_val=None,
        patience=7,
    ):
        self.hidden_size = hidden_size
        self.classes_ = None
        self.input_size = input_size
        self.model = None
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience

    # @pm.auto_serialize(
    #     model=pms.PytorchStateSerializer()
    # )

    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Starting training with parameters: lr={self.lr}, hidden_size={self.hidden_size}, dropout={self.dropout}"
        )
        # print(
        #     f"Starting training with parameters: lr={self.lr}, hidden_size={self.hidden_size}, dropout={self.dropout}"
        # )
        self.model = TARNet(
            input_size=self.input_size,
            shared_representation_size=self.hidden_size,
            outcome_representation_size=self.hidden_size,
            dropout=self.dropout,
        ).to(device)
        # print(next(self.model.parameters()).device)
        self.optimizer_class = self.optimizer_class(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.X_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(self.X_val, dtype=torch.float32),
                torch.tensor(self.y_val, dtype=torch.float32),
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=1e-5)

        avg_train_loss = 0.0
        avg_val_loss = 0.0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, t_batch, y_batch = (
                    X_batch.to(device),
                    X_batch[:, -1].to(device),
                    y_batch.to(device),
                )
                self.optimizer_class.zero_grad()
                outputs, t_hat, c_hat = self.model(X_batch, t_batch)
                # print(outputs.size(), y_batch.size())
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer_class.step()
                train_loss += loss.item()

                # if batch_idx % 100 == 0:
                #     logger.info(
                #         f"Epoch [{epoch+1}/{self.epochs}], Step [{batch_idx}/{len(train_loader)}], Train Loss: {loss.item():.4f}"
                #     )
                #     print(
                #         f"Epoch [{epoch+1}/{self.epochs}], Step [{batch_idx}/{len(train_loader)}], Train Loss: {loss.item():.4f}"
                #     )
            avg_train_loss = train_loss / len(train_loader)

            if self.X_val is not None:
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, t_batch, y_batch = (
                            X_batch.to(device),
                            X_batch[:, -1].to(device),
                            y_batch.to(device),
                        )
                        outputs, t_hat, c_hat = self.model(X_batch, t_batch)
                        # print(outputs)
                        loss = self.criterion(outputs, y_batch)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
                )
                # Call early stopping
                early_stopping(avg_val_loss, self.model)

                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            early_stopping.load_best_weights(self.model)
        # self.train_loss = avg_train_loss if "avg_train_loss" in locals() else None
        # self.val_loss = avg_val_loss if "avg_val_loss" in locals() else None

        # return self
    def predict(self, X):
        proba = self.predict_proba(X)
        return proba

    def predict_proba(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor, t_tensor = X_tensor.to(device), X_tensor[:, -1].to(device)
        with torch.no_grad():
            output, t_hat, y_hat = self.model(X_tensor, t_tensor)
            proba = torch.sigmoid(output).squeeze()
        return proba.cpu().numpy()

    def score(self, X, y):
        y_pred_proba = self.predict_proba(X)
        # logger.info(y_pred_proba)
        return roc_auc_score(y, y_pred_proba)

    def estimate_ite(self, X):
        """
        Estimate the Individual Treatment Effect (ITE) for each sample in X.

        Args:
            X (numpy.ndarray): The input features for each sample.

        Returns:
            numpy.ndarray: The estimated ITE for each sample.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info("No issues: 1")
        self.model.eval()
        # logger.info("No issues: 2")

        # Ensure treatment column exists and create two versions of X: one for treated and one for control
        assert "t" in X.columns, "X must include a 't' column for treatment assignment"
        # logger.info("No issues: 3")
        X_treated = X.copy()
        X_control = X.copy()
        X_treated["t"] = 1  # Set treatment to 1
        X_control["t"] = 0  # Set treatment to 0
        # logger.info("No issues: 4")

        # Convert to tensor
        X_treated_tensor = torch.tensor(X_treated.values, dtype=torch.float32).to(
            device
        )
        X_control_tensor = torch.tensor(X_control.values, dtype=torch.float32).to(
            device
        )
        # logger.info("No issues: 5")

        with torch.no_grad():
            # Predict outcomes under treatment and control
            _, treated_outcomes, _ = self.model(
                X_treated_tensor, X_treated_tensor[:, -1]
            )
            _, _, control_outcomes = self.model(
                X_control_tensor, X_control_tensor[:, -1]
            )
        # logger.info("No issues: 6")

        # Calculate ITE as the difference between treated and control predictions
        ite_estimates = treated_outcomes.cpu().numpy() - control_outcomes.cpu().numpy()
        treated = treated_outcomes.cpu().numpy().flatten()
        control = control_outcomes.cpu().numpy().flatten()
        return ite_estimates.squeeze(), treated, control  # Remove extra dimensions if they exist

    def decision_function(self, X):
        # For binary classification, decision_function can often be approximated
        # by the logit (log of odds) of the probability of the positive class.
        # This method assumes your predict_proba method returns probabilities for the positive class.
        proba = self.predict_proba(X)
        # Transform probabilities to a decision function score.
        # This is a simple logit function for binary classification.
        # Note: You might need to adjust this based on your specific output.
        decision_scores = np.log(proba / (1 - proba))
        return decision_scores

    def save_model(self, file_path):
        """Save the model's state dictionary to a file."""
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path, map_location="cpu"):
        """Load the model's state dictionary from a file."""
        self.model.load_state_dict(torch.load(file_path, map_location=map_location))


def create_bootstrap_df(stats, combination, model):
    return pd.DataFrame(
        {
            "ate": stats["ate_mean"],
            "ate_lower": stats["ate_lower"],
            "ate_upper": stats["ate_upper"],
            "auc": stats["auc_mean"],
            "auc_lower": stats["auc_lower"],
            "auc_upper": stats["auc_upper"],
            "drug_0": combination[0],
            "drug_1": combination[1],
            "model": model,
        },
        index=[0],
    )

def create_counterfactual_df(ids, y0, y1, treatment_drug):
    out = {}
    out["person_id"] = ids
    out["Y1"] = y1
    out["Y0"] = y0
    out["treatment_drug"] = treatment_drug
    return pd.DataFrame.from_dict(out)

def create_best_params_df(best_params, best_roc, best_ate, combination, model):
    best_params["val_roc"] = best_roc
    best_params["val_ate"] = best_ate
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


# def bootstrap_ate_estimates(
#     X, y, trained_lr, n_bootstraps=100, confidence_level=0.95, bootstrap_size=None
# ):
#     """
#     Estimate the Average Treatment Effect (ATE) and its confidence interval using bootstrapping.

#     This function resamples the original dataset with replacement to create multiple bootstrap samples,
#     computes the ATE for each sample using the provided model, and then calculates the average ATE
#     and its confidence interval from these estimates.

#     Args:
#         X (numpy.ndarray): The input features for each sample.
#         y (numpy.ndarray): The output labels for each sample.
#         best_model (PyTorchClassifierWrapper): A trained instance of the TARNet model or similar,
#                                                 used to estimate the ATE for each bootstrap sample.
#         n_bootstraps (int, optional): The number of bootstrap samples to generate. Defaults to 1000.
#         confidence_level (float, optional): The confidence level for the confidence interval of the ATE.
#                                             Should be between 0 and 1. Defaults to 0.95.
#         bootstrap_size (int, optional): The size of each bootstrap sample. If None, the bootstrap sample
#                                         will be the same size as the original dataset. Defaults to None.

#     Returns:
#         float: The average estimated ATE across all bootstrap samples.
#         numpy.ndarray: The confidence interval for the ATE, given the specified confidence level.

#     Raises:
#         ValueError: If `confidence_level` is not between 0 and 1.
#     """
#     if not (0 < confidence_level < 1):
#         raise ValueError("confidence_level must be between 0 and 1")

#     ate_estimates = []
#     n_samples = X.shape[0] if bootstrap_size is None else bootstrap_size
#     for i in range(n_bootstraps):
#         # logger.info(f"Running iteration {i+1} of {n_bootstraps}")
#         # Resample the data with replacement, using the specified bootstrap_size
#         X_resampled, y_resampled = resample(X, y, n_samples=n_samples)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         class_weights = class_weight.compute_class_weight(
#             class_weight="balanced", classes=np.unique(y), y=y_resampled
#         )
#         class_weight_dict = dict(enumerate(class_weights))
#         class_weights_tensor = torch.tensor(
#             [class_weight_dict[0], class_weight_dict[1]], dtype=torch.float32
#         )

#         class_weights_tensor = class_weights_tensor.to(device)

#         wrapper = PyTorchClassifierWrapper(
#             input_size=X.shape[1],  # Default values, will be overridden by GridSearchCV
#             hidden_size=256,  # Default values, will be overridden by GridSearchCV
#             lr=trained_lr,  # Default values, will be overridden by GridSearchCV
#             batch_size=128,  # Default values, will be overridden by GridSearchCV
#             epochs=50,  # Default values, will be overridden by GridSearchCV
#             dropout=0.1,  # Default values, will be overridden by GridSearchCV
#             criterion=nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1]),
#             optimizer_class=optim.Adam,
#             # X_val=X_val.values,  # Pass validation data here
#             # y_val=y_val.values,  # Pass validation data here
#             patience=5,
#         )

#         wrapper.fit(X_resampled.values, y_resampled.values)
#         # Recompute the ITE estimates on the resampled dataset using the best model found
#         ite_estimates, _, _ = wrapper.estimate_ite(X_resampled)
#         # Compute the Average Treatment Effect (ATE) for this bootstrap sample
#         ate_estimates.append(ite_estimates.mean())

#     # Compute the percentile intervals for the ATE
#     lower_percentile = ((1 - confidence_level) / 2) * 100
#     upper_percentile = (1 - (1 - confidence_level) / 2) * 100
#     lower_bound = np.percentile(ate_estimates, lower_percentile)
#     upper_bound = np.percentile(ate_estimates, upper_percentile)

#     return np.mean(ate_estimates), lower_bound, upper_bound
def single_bootstrap_iteration(X, y, trained_lr, bootstrap_size, device):
    # Resample the data with replacement
    X_resampled, y_resampled = resample(X, y, n_samples=bootstrap_size)

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y_resampled
    )
    class_weight_dict = dict(enumerate(class_weights))
    class_weights_tensor = torch.tensor(
        [class_weight_dict[0], class_weight_dict[1]], dtype=torch.float32
    ).to(device)

    wrapper = PyTorchClassifierWrapper(
        input_size=X.shape[1],
        hidden_size=256,
        lr=trained_lr,
        batch_size=128,
        epochs=50,
        dropout=0.1,
        criterion=nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1]),
        optimizer_class=optim.Adam,
        patience=5,
    )

    wrapper.fit(X_resampled.values, y_resampled.values)

    ite, _, _ = wrapper.estimate_ite(X_resampled)
    ate = ite.mean()

    y_pred_proba = wrapper.predict_proba(X_resampled.values)
    auc = roc_auc_score(y_resampled.values, y_pred_proba)

    return ate, auc


def bootstrap_ate_estimates(
    X, y, trained_lr, n_bootstraps=100, confidence_level=0.95, bootstrap_size=None
):
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")

    n_samples = X.shape[0] if bootstrap_size is None else bootstrap_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parallel execution of bootstrap iterations
    results = Parallel(n_jobs=2)(
        delayed(single_bootstrap_iteration)(X, y, trained_lr, n_samples, device)
        for _ in range(n_bootstraps)
    )

    ates, aucs = zip(*results)
    ates = np.array(ates)
    aucs = np.array(aucs)

    def ci(arr):
        lower = np.percentile(arr, 100 * (1 - confidence_level) / 2)
        upper = np.percentile(arr, 100 * (1 + confidence_level) / 2)
        return arr.mean(), lower, upper

    ate_mean, ate_lower, ate_upper = ci(ates)
    auc_mean, auc_lower, auc_upper = ci(aucs)

    return {
        "ate_mean": ate_mean,
        "ate_lower": ate_lower,
        "ate_upper": ate_upper,
        "auc_mean": auc_mean,
        "auc_lower": auc_lower,
        "auc_upper": auc_upper,
    }

def bootstrap_func(main_df, hyperparams_df, ingredient_pairs):
    results_df = pd.DataFrame()
    for idx, combination in enumerate(ingredient_pairs):
        # start_time = datetime.now()
        logger.info(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )
        # print(
        #     f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        # )
        df = main_df.loc[main_df.ingredient_concept_id.isin(list(combination))]
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        X = df.drop(
            ["person_id", "severity_final", "ingredient_concept_id", "treatment"],
            axis=1,
        )
        y = df["severity_final"]
        X["t"] = df["treatment"]

        np.random.seed(100)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        np.random.seed(100)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=y_train_val,
        )

        logger.info("Running bootstrapping to find the confidence interval")
        try:
            trained_lr = hyperparams_df[
                (hyperparams_df["drug_0"] == combination[0])
                & (hyperparams_df["drug_1"] == combination[1])
            ]["lr"].values[0]
        except Exception as e:
            try:
                trained_lr = hyperparams_df[
                    (hyperparams_df["drug_1"] == combination[1])
                    & (hyperparams_df["drug_0"] == combination[0])
                ]["lr"].values[0]
            except Exception as e:
                dicttmp = {
                    "comb1": [str(combination[0])],
                    "comb2": [str(combination[1])],
                    "error": [str(e)]
                }
                return pd.DataFrame(dicttmp)
        logger.info(f"Training the model with params: learning rate: {trained_lr}")
        print(f"Training the model with params: learning rate: {trained_lr}")

        ate_mean, ate_lower, ate_upper = bootstrap_ate_estimates(
            X_train_val,
            y_train_val,
            trained_lr,
            n_bootstraps=100,
            confidence_level=0.95,
            bootstrap_size=X_train_val.shape[0],
        )

        variance = ate_upper - ate_lower

        bootstrap_df = create_bootstrap_df(
            ate_mean, ate_lower, ate_upper, variance, combination, "TARNet"
        )

        results_df = pd.concat([results_df, bootstrap_df], ignore_index=True)

    return results_df


def tarnet_func(main_df, ingredient_pairs):
    # Use environment - RP-1225E6 for this notebook.
    # Note: TensorFlow and CausalML cannot be run in the same script or environment due to system dependencies.
    # This workbook is used only for TensorFlow.

    # Create and get the data for pair of different antidepressants
    # main_df = main_df.toPandas()
    results_df = pd.DataFrame()
    output_counterfactual = pd.DataFrame()
    # ingredient_list = main_df.ingredient_concept_id.unique()
    # ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

    params = {
        # "lr": [10 ** np.random.uniform(-5, -2) for _ in range(5)],
        "lr": [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2],
        # "dropout": [0, 0.2],
    }
    # params = {
    # #     # "lr": [10 ** np.random.uniform(-5, -2) for _ in range(5)],
    #     "lr": [1e-3],
    # #     # "dropout": [0, 0.2],
    # }
    roc_auc_scorer = make_scorer(
        roc_auc_score, greater_is_better=True
    )

    # Define KFold for cross-validation
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for idx, combination in enumerate(ingredient_pairs):
        # start_time = datetime.now()
        logger.info(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )
        # print(
        #     f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        # )
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        X = df.drop(
            ["severity_final", "ingredient_concept_id", "treatment"],
            axis=1,
        )
        y = df["severity_final"]
        X["t"] = df["treatment"]
        
        np.random.seed(100)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )   
        np.random.seed(100)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=y_train_val,
        )
        people = X_test["person_id"]
        X = X.drop(["person_id"], axis=1)
        X_train_val = X_train_val.drop(["person_id"], axis=1)
        X_test = X_test.drop(["person_id"], axis=1)
        X_train = X_train.drop(["person_id"], axis=1)
        X_val = X_val.drop(["person_id"], axis=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        class_weights_tensor = torch.tensor(
            [class_weight_dict[0], class_weight_dict[1]], dtype=torch.float32
        )

        class_weights_tensor = class_weights_tensor.to(device)

        logger.info("Running GridSearch to find the best parameters")
        wrapper = PyTorchClassifierWrapper(
            input_size=X.shape[1],  # Default values, will be overridden by GridSearchCV
            hidden_size=256,  # Default values, will be overridden by GridSearchCV
            lr=0.001,  # Default values, will be overridden by GridSearchCV
            batch_size=128,  # Default values, will be overridden by GridSearchCV
            epochs=50,  # Default values, will be overridden by GridSearchCV
            dropout=0.1,  # Default values, will be overridden by GridSearchCV
            criterion=nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1]),
            optimizer_class=optim.Adam,
            X_val=X_val.values,  # Pass validation data here
            y_val=y_val.values,  # Pass validation data here
            patience=5,
        )

        # Set up GridSearchCV
        grid = GridSearchCV(
            estimator=wrapper,
            param_grid=params,
            cv=folds,
            scoring=roc_auc_scorer,
            # verbose=3,
            error_score="raise"
        )

        # Fit the model
        grid.fit(X_train.values, y_train.values)
        best_score = grid.best_score_  # Assuming X and y are numpy arrays
        best_model = grid.best_estimator_
        ite, treatment, control = best_model.estimate_ite(X_test)
        ate = ite.mean()

        logger.info("Training done!")
        logger.info(f"Best ROC {best_score} for parameters: {grid.best_params_}")
        logger.info(f"ATE: {ate}")

        # Saving the best model parameters and the metadata
        file_name = f"{combination[0]}_{combination[1]}.pth"
        best_model.save_model(file_name)
        # with open(file_name, "wb") as f:
        #     pickle.dump(grid, f)

        best_params_df = create_best_params_df(
            grid.best_params_, best_score, ate, combination, "TARNet"
        )

        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        counterfactual_df = create_counterfactual_df(people, treatment, control, X_test["t"])
        counterfactual_df["drug_0"] = combination[0]
        counterfactual_df["drug_1"] = combination[1]
        counterfactual_df["model"] = "Tarnet"
        output_counterfactual = pd.concat([output_counterfactual, counterfactual_df], ignore_index=True)
        del grid
        del wrapper
        del best_params_df

    return results_df, output_counterfactual


def bootstrapping(X_test, y_test, best_model, n_bootstraps, bootstrap_size):
    # Bootstrapping for ATE confidence interval with specified bootstrap sample size
    ate_mean, ate_lower, ate_upper = bootstrap_ate_estimates(
        X_test,
        y_test,
        best_model,
        n_bootstraps=n_bootstraps,
        confidence_level=0.95,
        bootstrap_size=bootstrap_size,
    )


# @configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
# @configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_SMALL"])
# @transform(
#     output_df=Output("ri.foundry.main.dataset.efd8f778-447f-42b2-a2e1-f9b266d874fc"),
#     source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
# )
# def compute(output_df, source_df):
#     source_df = source_df.dataframe()

#     source_df = source_df.toPandas()
#     results_df = pd.DataFrame()
#     ingredient_list = source_df.ingredient_concept_id.unique()
#     ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

#     mini_batch_size = 5
#     mini_batches = []

#     for idx in range(0, len(ingredient_pairs), mini_batch_size):
#         # print(idx, idx+mini_batch_size)
#         mini_batches.append(ingredient_pairs[idx : idx + mini_batch_size])

#     with Pool(processes=8) as pool:
#         # prepare arguments
#         items = [(source_df, batch) for batch in mini_batches]
#         # execute tasks and process results in order
#         for result in pool.starmap(wrapped_func, items):
#             print(f'Got result: {result}')
#             # break

#     # pool = Pool(processes=len(mini_batches))
#     # pool.starmap(tarnet_func, [(source_df, batch) for batch in mini_batches])

#     # results_df = tarnet(source_df)
#     logger.info(type(results_df))

#     # Create a SparkSession object
#     spark = SparkSession.builder.getOrCreate()
#     logger.info("No issues till now")
#     # print("No issues till now")

#     # Convert the Pandas DataFrame to a PySpark DataFrame
#     df_pyspark = spark.createDataFrame(results_df)
#     logger.info("No issues till now")
#     # print(f"No issues till now. Type of df {type(df_pyspark)}")

#     return output_df.write_dataframe(df_pyspark)
