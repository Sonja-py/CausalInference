from pyspark.sql.functions import col, lit, concat, explode

# from pyspark.sql import SparkSession
# from transforms.api import transform, Input, Output, configure
import pandas as pd
import numpy as np

# from itertools import combinations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    ParameterGrid,
)
from sklearn.utils import resample

from sklearn.base import BaseEstimator, ClassifierMixin

from torch_geometric.nn import GCNConv
import random
import copy

from torch.utils.tensorboard import SummaryWriter



logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)


def create_best_params_df(best_params, test_roc, val_roc, y1, y0, test_ate, combination, model):
    best_params["test_roc"] = test_roc
    best_params["val_roc"] = val_roc
    best_params["y1"] = y1
    best_params["y0"] = y0
    best_params["test_ate"] = test_ate
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


def create_bootstrap_df(ate, ate_lower, ate_upper, variance, combination, model):
    best_params = {}
    best_params["ate"] = ate
    best_params["ate_lower"] = ate_lower
    best_params["ate_upper"] = ate_upper
    best_params["variance"] = variance
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


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


class NetDeconf(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False):
        """
        The NetDeconf class is a PyTorch module that implements a neural network for deconfounding using graph convolutional networks.

        Args:
            nfeat (int): The number of input features.
            nhid (int): The number of hidden units.
            dropout (float): The dropout rate.
            n_in (int, optional): The number of input layers. Defaults to 1.
            n_out (int, optional): The number of output layers. Defaults to 1.
            cuda (bool, optional): Whether to use CUDA. Defaults to False.
        """
        super(NetDeconf, self).__init__()

        device = torch.device("cuda" if cuda else "cpu")

        # Create a list of GCNConv layers
        self.gc = nn.ModuleList(
            [GCNConv(nfeat, nhid)] + [GCNConv(nhid, nhid) for _ in range(n_in - 1)]
        )
        self.gc = self.gc.to(device)  # Move the layers to the device

        self.n_in = n_in  # Number of input layers
        self.n_out = n_out  # Number of output layers

        self.out_t00 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )  # Linear layer for the output of the control
        self.out_t10 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )  # Linear layer for the output of the treatment
        self.out_t01 = nn.Linear(nhid, 1).to(
            device
        )  # Linear layer for the output of the control
        self.out_t11 = nn.Linear(nhid, 1).to(
            device
        )  # Linear layer for the output of the treatment

        self.dropout = dropout  # Dropout rate
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, data, patient_ids, treatment=None):
        """
        The forward method is used to compute the output of the neural network.

        Args:
            data (Data): The input PyTorch Geometric data.
            patient_ids (_type_): The patient IDs.
            treatment (tensor): The treatment values for patient_ids

        Returns:
            _type_: _description_
        """
        x, adj = (
            data.x,
            data.edge_index,
        )  # Get the features and the edge index (adjacency matrix)

        dist = F.relu(self.gc[0](x, adj))  # Apply the first GCNConv layer
        dist = F.dropout(dist, self.dropout, training=self.training)  # Apply dropout

        for layer in range(1, self.n_in):  # Apply the rest of the GCNConv layers
            dist = F.relu(
                self.gc[layer](dist, adj)
            )  # Apply the GCNConv layer at index layer
            dist = F.dropout(
                dist, self.dropout, training=self.training
            )  # Apply dropout

        for layer in range(
            self.n_out
        ):  # Apply the output layers for the control and treatment
            y00 = F.relu(
                self.out_t00[layer](dist)
            )  # Apply the output layer for the control
            y00 = F.dropout(y00, self.dropout, training=self.training)  # Apply dropout
            y10 = F.relu(
                self.out_t10[layer](dist)
            )  # Apply the output layer for the treatment
            y10 = F.dropout(y10, self.dropout, training=self.training)  # Apply dropout

        y0 = self.sigmoid(
            self.out_t01(y00).view(-1)
        )  # Apply the output layer for the control and reshape
        y1 = self.sigmoid(
            self.out_t11(y10).view(-1)
        )  # Apply the output layer for the treatment and reshape

        # Get the output for the control or treatment only for the patient_ids
        y0 = y0[patient_ids]
        y1 = y1[patient_ids]

        # y = torch.where(data.t[patient_ids] > 0, y1, y0)  # Get the output for the control or treatment based on the treatment
        y = torch.where(
            treatment > 0, y1, y0
        )  # Get the output for the control or treatment based on the treatment

        return y, y1, y0  # Return the output


class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        patient_ids,
        treatment,
        hidden_dim,
        n_in,
        n_out,
        criterion,
        optimizer_class,
        lr,
        epochs,
        dropout,
        patience=7,
    ):
        self.patient_ids = patient_ids
        self.treatment = treatment
        self.hidden_dim = hidden_dim
        self.n_in = n_in
        self.n_out = n_out
        self.model = None
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.patience = patience

    def fit(self, data, outcome):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Starting training with parameters: lr={self.lr}, hidden_size={self.hidden_dim}, n_in={self.n_in}, n_out={self.n_out}"
        )
        # print(
        #     f"Starting training with parameters: lr={self.lr}, hidden_size={self.hidden_size}, dropout={self.dropout}"
        # )
        self.model = NetDeconf(
            nfeat=data.x.shape[1],
            nhid=self.hidden_dim,
            dropout=self.dropout,
            n_in=self.n_in,
            n_out=self.n_out,
            cuda=False,
        ).to(device)
        # print(next(self.model.parameters()).device)
        self.optimizer_class = self.optimizer_class(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=1e-5)

        avg_val_loss = 0.0

        for epoch in range(self.epochs):
            # Training phase

            data = data.to(device)
            self.model.train()
            train_loss = 0.0
            self.optimizer_class.zero_grad()
            output, _, _ = self.model(
                data, self.patient_ids["train"].values, self.treatment["train"]
            )

            train_loss = self.criterion(output, outcome["train"])
            train_loss.backward()
            self.optimizer_class.step()

            # Validation phase
            total_loss = 0
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                data = data.to(device)
                output, _, _ = self.model(
                    data,
                    self.patient_ids["validation"].values,
                    self.treatment["validation"],
                )
                loss = self.criterion(output, outcome["validation"])
                total_loss += loss.item()

                # Compute the accuracy
                predicted = output.round()
                correct += (predicted == self.outcome["validation"]).sum().item()
                total += len(self.patient_ids["validation"])

            val_loss = total_loss / total

            logger.info(
                f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
            )

            # Call early stopping
            early_stopping(avg_val_loss, self.model)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        early_stopping.load_best_weights(self.model)

    def predict_proba(self, data, patient_ids, treatment, outcome):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        data = data.to(device)
        # X_tensor = torch.tensor(X, dtype=torch.float32)
        # X_tensor, t_tensor = X_tensor.to(device), X_tensor[:, -1].to(device)
        with torch.no_grad():
            output, _, _ = self.model(data, patient_ids, treatment, outcome)
            proba = torch.sigmoid(output)
        return proba.cpu().numpy()

    def score(self, data, patient_ids, treatment, outcome):
        y_pred_proba = self.predict_proba(data, patient_ids, treatment, outcome)
        # logger.info(y_pred_proba)
        return roc_auc_score(outcome, y_pred_proba)

    def estimate_ate(self, data, patient_ids, treatment):
        """
        Estimate the Average Treatment Effect (ATE) for each sample in X.

        Args:
            X (numpy.ndarray): The input features for each sample.

        Returns:
            numpy.ndarray: The estimated ATE for each sample.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            data = data.to(device)
            _, y1, y0 = self.model(data, patient_ids["test"].values, treatment["test"])
            ate = torch.mean(y1 - y0).item()

        return ate, y1, y0

    def decision_function(self, data, patient_ids, treatment, outcome):
        # For binary classification, decision_function can often be approximated
        # by the logit (log of odds) of the probability of the positive class.
        # This method assumes your predict_proba method returns probabilities for the positive class.
        proba = self.predict_proba(data, patient_ids, treatment, outcome)
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


def train(data, model, optimizer, criterion, patient_ids, treatment, outcome):
    # Set the model to training mode
    data = data.to(device)
    model.train()
    optimizer.zero_grad()
    output, _, _ = model(data, patient_ids, treatment)

    loss = criterion(output, outcome)
    loss.backward()
    optimizer.step()

    return loss


def evaluate(data, model, optimizer, criterion, patient_ids, treatment, outcome):
    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output, _, _ = model(data, patient_ids, treatment)
        loss = criterion(output, outcome)
        total_loss += loss.item()

        # Compute the accuracy
        predicted = output.round()
        correct += (predicted == outcome).sum().item()
        total += len(patient_ids)

    val_loss = total_loss / total
    val_acc = correct / total

    return val_loss, val_acc


def roc_auc(data, model, patient_ids, treatment, outcome):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output, y1, y0 = model(data, patient_ids, treatment)
        roc = roc_auc_score(outcome.cpu(), output.cpu())

    return roc, y1, y0


def estimate_ate(data, model, patient_ids, treatment):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        _, y1, y0 = model(data, patient_ids, treatment)
        ate = torch.mean(y1 - y0).item()
    return ate


def bootstrap_ate_estimates(
    data,
    patient_severity,
    patient_node_indices,
    treatment,
    outcome,
    params,
    n_bootstraps=100,
    confidence_level=0.95,
    bootstrap_size=None,
):
    """
    Estimate the Average Treatment Effect (ATE) and its confidence interval using bootstrapping.

    This function resamples the original dataset with replacement to create multiple bootstrap samples,
    computes the ATE for each sample using the provided model, and then calculates the average ATE
    and its confidence interval from these estimates.

    Args:
        X (numpy.ndarray): The input features for each sample.
        y (numpy.ndarray): The output labels for each sample.
        best_model (PyTorchClassifierWrapper): A trained instance of the TARNet model or similar,
                                                used to estimate the ATE for each bootstrap sample.
        n_bootstraps (int, optional): The number of bootstrap samples to generate. Defaults to 1000.
        confidence_level (float, optional): The confidence level for the confidence interval of the ATE.
                                            Should be between 0 and 1. Defaults to 0.95.
        bootstrap_size (int, optional): The size of each bootstrap sample. If None, the bootstrap sample
                                        will be the same size as the original dataset. Defaults to None.

    Returns:
        float: The average estimated ATE across all bootstrap samples.
        numpy.ndarray: The confidence interval for the ATE, given the specified confidence level.

    Raises:
        ValueError: If `confidence_level` is not between 0 and 1.
    """
    num_epochs = 100
    random.seed(42)
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")

    ate_estimates = []
    n_samples = (
        patient_node_indices.shape[0] if bootstrap_size is None else bootstrap_size
    )
    for i in range(n_bootstraps):
        logger.info(f"Running iteration {i+1} of {n_bootstraps}")
        # Resample the data with replacement, using the specified bootstrap_size
        patient_node_resampled = resample(
            patient_node_indices, replace=True, n_samples=n_samples, random_state=42
        )
        treatment_resampled = patient_severity.loc[
            patient_node_resampled
        ].treatment.values
        outcome_resampled = patient_severity.loc[
            patient_node_resampled
        ].severity_final.values

        treatment_resampled = torch.tensor(treatment_resampled, dtype=torch.float).to(
            device
        )
        outcome_resampled = torch.tensor(outcome_resampled, dtype=torch.float).to(
            device
        )

        model = NetDeconf(
            nfeat=data.x.shape[1],
            nhid=params["hidden_dim"],
            dropout=0.1,
            n_in=params["n_in"],
            n_out=params["n_out"],
            cuda=False,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=5e-4
        )
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            train_loss = train(
                data,
                model,
                optimizer,
                criterion,
                patient_node_resampled,
                treatment_resampled,
                outcome_resampled,
            )
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")
            logger.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

        # Calculate ATE for the best model on the test set
        ate = estimate_ate(data, model, patient_node_resampled, treatment_resampled)
        ate_estimates.append(ate)

    # Compute the percentile intervals for the ATE
    lower_percentile = ((1 - confidence_level) / 2) * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100
    lower_bound = np.percentile(ate_estimates, lower_percentile)
    upper_bound = np.percentile(ate_estimates, upper_percentile)

    return np.mean(ate_estimates), lower_bound, upper_bound


def net_deconf_main(data, patient_severity, treatment, outcome):
    num_epochs = 100
    # param_grid = {
    #     "lr": [1e-5, 1e-4, 1e-3, 1e-2, 5e-5, 5e-4, 5e-3, 5e-2],
    #     "hidden_dim": [4, 8, 16, 32, 128, 256, 512],
    #     "n_in": [1, 2, 3, 4, 8, 16],
    #     "n_out": [1, 2, 3, 4, 8, 16],
    # }
    param_grid = {
        "lr": [1e-4],
        "hidden_dim": [128],
        "n_in": [2],
        "n_out": [2],
    }
    

    num_nodes = data.num_nodes

    indices = torch.arange(num_nodes)
    values = torch.ones(num_nodes)
    
    ### Temporarily removed for testing learning
    # node_features = torch.sparse_coo_tensor(
    #     indices=torch.vstack([indices, indices]),
    #     values=values,
    #     size=(num_nodes, num_nodes),
    # )

    node_features = torch.rand(num_nodes, 100)
    data.x = node_features

    logger.info(data)

    # Define KFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random.seed(42)
    patient_node_indices = patient_severity.index.values

    roc_auc_scorer = make_scorer(
        roc_auc_score, greater_is_better=True, needs_threshold=True
    )

    # Create train, val, test splits
    (
        train_val_patient_ids,
        test_patient_ids,
        train_val_treatment,
        test_treatment,
        train_val_outcome,
        test_outcome,
    ) = train_test_split(
        patient_node_indices, treatment, outcome, test_size=0.2, random_state=42
    )

    train_val_treatment = torch.tensor(train_val_treatment.values, dtype=torch.float)
    test_treatment = torch.tensor(test_treatment.values, dtype=torch.float).to(device)
    train_val_outcome = torch.tensor(train_val_outcome.values, dtype=torch.float)
    test_outcome = torch.tensor(test_outcome.values, dtype=torch.float).to(device)

    best_model = None
    best_roc = -np.Inf
    best_params = None

    random.seed(42)
    # tensorboard_outs = []

    out_loss = dict()
    current_loss = dict()
    for params in ParameterGrid(param_grid):
        logger.info(f"Training with params: {params}")
        fold_scores = []
        criterion = nn.BCELoss()
        fold = 0
        for train_idx, val_idx in skf.split(train_val_patient_ids, train_val_treatment):
            fold += 1
            train_patient_ids, val_patient_ids = (
                train_val_patient_ids[train_idx],
                train_val_patient_ids[val_idx],
            )
            train_treatment, val_treatment = train_val_treatment[train_idx].to(
                device
            ), train_val_treatment[val_idx].to(device)
            train_outcome, val_outcome = train_val_outcome[train_idx].to(
                device
            ), train_val_outcome[val_idx].to(device)

            model = NetDeconf(
                nfeat=data.x.shape[1],
                nhid=params["hidden_dim"],
                dropout=0.1,
                n_in=params["n_in"],
                n_out=params["n_out"],
                cuda=False,
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params["lr"], weight_decay=5e-4
            )
            # early_stopping = EarlyStopping(patience=50, verbose=True, delta=1e-6)
            # x = "ri.compass.main.folder.cd88afc9-131e-4c6b-8b11-10ce14c54096"
            # log_dir = f'../workbook-output/Network Deconfounder/Tensorboards/Fold_{fold}_{params}'
            # swriter = SummaryWriter(log_dir=log_dir)
            losses = []
            for epoch in range(num_epochs):
                train_loss = train(
                    data,
                    model,
                    optimizer,
                    criterion,
                    train_patient_ids,
                    train_treatment,
                    train_outcome,
                )
                val_loss, _ = evaluate(
                    data,
                    model,
                    optimizer,
                    criterion,
                    val_patient_ids,
                    val_treatment,
                    val_outcome,
                )
                print(
                    f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
                )
                logger.info(
                    f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
                )
                # swriter.add_scalar('Loss/Val', val_loss, epoch)
                # swriter.add_scalar('Loss/Train', train_loss, epoch)
                losses.append(float(train_loss.item()))
                # losses["val"] = float(val_loss)
            # swriter.close()

                # early_stopping(val_loss, model)
                # if early_stopping.early_stop:
                #     logger.info(f"Early stopping at epoch {epoch+1}")
                #     break
            current_loss[fold] = losses
            
            # early_stopping.load_best_weights(model)
            roc_scores, _, _ = roc_auc(
                data, model, val_patient_ids, val_treatment, val_outcome
            )
            fold_scores.append(roc_scores)

        mean_score = np.mean(fold_scores)
        logger.info(f"Mean ROC AUC score for params {params}: {mean_score}")

        # Save the best_params
        if mean_score > best_roc:
            best_roc = mean_score
            best_model = copy.deepcopy(model)
            best_params = params
            out_loss = current_loss
        current_loss = {}

        logger.info(f"Best params: {best_params}")
        logger.info(f"Best ROC AUC score: {best_roc}")

    # Calculate ATE for the best model on the test set
    ate, y1, y0 = estimate_ate(data, best_model, test_patient_ids, test_treatment)
    test_roc = roc_auc(data, best_model, test_patient_ids, test_treatment, test_outcome)

    return test_roc, best_roc, y1, y0, ate, best_params, out_loss


def bootstrap_func(
    main_df,
    hyperparams_df,
    ingredient_pairs,
    person_condition_network,
    person_drug_network,
    drug_condition_network,
    drug_drug_network,
    condition_condition_network,
    condition_concept_network,
):
    results_df = pd.DataFrame()
    for idx, combination in enumerate(ingredient_pairs):
        print(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )
        logger.info(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )

        filter_values_list = [int(x) for x in list(combination)]
        patient_severity = main_df.toPandas().copy()
        patient_severity["treatment"] = patient_severity["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        patient_severity = patient_severity[
            patient_severity["ingredient_concept_id"].isin(list(combination))
        ]
        patient_severity = patient_severity[
            ["person_id", "treatment", "severity_final"]
        ]
        patient_severity["person_id"] = patient_severity["person_id"].apply(
            lambda x: f"person_{x}"
        )

        person_condition = person_condition_network.alias("person_condition")
        person_drug = person_drug_network.alias("person_drug")
        drug_condition = drug_condition_network.alias("drug_condition")
        drug_drug = drug_drug_network.alias("drug_drug")
        condition_condition = condition_condition_network.alias("condition_condition")
        condition_concept = condition_concept_network.alias("condition_concept")

        person_condition = (
            person_condition.withColumn("condition", explode("condition_agg"))
            .withColumn("ingredient_concept_id", explode("ingredient_agg"))
            .select("person_id", "condition", "ingredient_concept_id")
            .dropDuplicates()
        )

        person_condition = (
            person_condition.withColumn(
                "node_1", concat(lit("person_"), col("person_id"))
            )
            .withColumn("node_2", concat(lit("condition_"), col("condition")))
            .filter(col("ingredient_concept_id").isin(filter_values_list))
            .withColumn("type", lit("person-condition"))
            .select("node_1", "node_2")
            .toPandas()
        )

        person_drug = (
            person_drug.withColumn("node_2", explode("node_2_agg"))
            .withColumn("ingredient_concept_id", explode("ingredient_agg"))
            .select("node_1", "node_2", "ingredient_concept_id")
        )

        person_drug = (
            person_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
            .select("node_1", "node_2")
            .toPandas()
        )

        drug_drug = (
            drug_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
            .select("node_1", "node_2")
            .toPandas()
        )

        condition_condition = condition_condition.select("node_1", "node_2").toPandas()

        drug_condition = (
            drug_condition.withColumn(
                "node_1", concat(lit("drug_"), col("drug_rxnorm_concept_id"))
            )
            .withColumn(
                "node_2", concat(lit("condition_"), col("condition_snomed_concept_id"))
            )
            .select("node_1", "node_2")
            .toPandas()
        )

        condition_concept = (
            condition_concept.withColumn(
                "node_1", concat(lit("condition_"), col("index_prefix"))
            )
            .withColumn("node_2", concat(lit("concept_"), col("concept_id")))
            .withColumn("type", lit("condition-condition_concept"))
            .select("node_1", "node_2")
            .toPandas()
        )

        # Compile all identifiers for each type into unified lists
        all_patient_ids = pd.concat(
            [person_drug["node_1"], person_condition["node_1"]]
        ).unique()

        patient_severity = patient_severity[
            patient_severity["person_id"].isin(all_patient_ids)
        ].reset_index(drop=True)

        all_drug_ids = pd.concat(
            [
                person_drug["node_2"],
                drug_drug["node_1"],
                drug_drug["node_2"],
                drug_condition["node_1"],
            ]
        ).unique()
        all_condition_ids = pd.concat(
            [
                person_condition["node_2"],
                condition_condition["node_1"],
                condition_condition["node_2"],
                drug_condition["node_2"],
                condition_concept["node_2"],
            ]
        ).unique()
        all_condition_classes = condition_concept["node_1"].unique()

        # Create global mappings with unique indices across all types
        global_index = 0
        patient_id_to_index = {}
        drug_id_to_index = {}
        condition_id_to_index = {}
        condition_class_to_index = {}

        # Assign indices for patients
        for patient in all_patient_ids:
            patient_id_to_index[patient] = global_index
            global_index += 1

        # Assign indices for drugs, offset by the number of patients
        for drug in all_drug_ids:
            drug_id_to_index[drug] = global_index
            global_index += 1

        for condition in all_condition_ids:
            condition_id_to_index[condition] = global_index
            global_index += 1

        for condition_class in all_condition_classes:
            condition_class_to_index[condition_class] = global_index
            global_index += 1

        # Apply mappings to data frames
        person_drug["node_1"] = person_drug["node_1"].map(patient_id_to_index)
        person_drug["node_2"] = person_drug["node_2"].map(drug_id_to_index)

        person_condition["node_1"] = person_condition["node_1"].map(patient_id_to_index)
        person_condition["node_2"] = person_condition["node_2"].map(
            condition_id_to_index
        )

        drug_drug["node_1"] = drug_drug["node_1"].map(drug_id_to_index)
        drug_drug["node_2"] = drug_drug["node_2"].map(drug_id_to_index)

        drug_condition["node_1"] = drug_condition["node_1"].map(drug_id_to_index)
        drug_condition["node_2"] = drug_condition["node_2"].map(condition_id_to_index)

        condition_condition["node_1"] = condition_condition["node_1"].map(
            condition_id_to_index
        )
        condition_condition["node_2"] = condition_condition["node_2"].map(
            condition_id_to_index
        )

        condition_concept["node_1"] = condition_concept["node_1"].map(
            condition_class_to_index
        )
        condition_concept["node_2"] = condition_concept["node_2"].map(
            condition_id_to_index
        )

        patient_severity["person_id"] = patient_severity["person_id"].map(
            patient_id_to_index
        )
        # patient_ids = patient_severity["person_id"]
        all_edges = pd.concat(
            [
                person_drug,
                person_condition,
                drug_drug,
                drug_condition,
                condition_condition,
                condition_concept,
            ]
        )

        # Get the (patient_id, treatment) pairs for the patients who have received both treatments
        selected_pairs = (
            patient_severity.groupby("person_id")
            .filter(lambda x: x["treatment"].nunique() > 1)
            .sort_values(by="person_id")
        )

        # From each unique (patient_id, treatment), randomly select one row
        selected_pairs = (
            selected_pairs.groupby("person_id")
            .apply(lambda x: x.sample(1))
            .reset_index(drop=True)
        )

        # Make a list of (patient_id, treatment) pairs
        pairs_to_drop = list(
            zip(selected_pairs["person_id"], selected_pairs["treatment"])
        )

        # Drop the selected pairs from the dataframe
        patient_severity = patient_severity[
            ~patient_severity.apply(
                lambda x: (x["person_id"], x["treatment"]) in pairs_to_drop, axis=1
            )
        ].reset_index(drop=True)

        # patient_ids = patient_severity['person_id']
        patient_severity = patient_severity.set_index("person_id")
        treatment = patient_severity["treatment"]
        outcome = patient_severity["severity_final"]

        edge_index = torch.tensor(all_edges.values, dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(edge_index.size(1))

        data = Data(
            num_nodes=global_index, edge_index=edge_index, edge_attr=edge_weight
        )

        num_nodes = data.num_nodes

        indices = torch.arange(num_nodes)
        values = torch.ones(num_nodes)
        node_features = torch.sparse_coo_tensor(
            indices=torch.vstack([indices, indices]),
            values=values,
            size=(num_nodes, num_nodes),
        )
        data.x = node_features

        logger.info(data)

        patient_node_indices = patient_severity.index.values
        # Create train, val, test splits
        (
            train_val_patient_ids,
            test_patient_ids,
            train_val_treatment,
            test_treatment,
            train_val_outcome,
            test_outcome,
        ) = train_test_split(
            patient_node_indices, treatment, outcome, test_size=0.2, random_state=42
        )

        train_val_treatment = torch.tensor(
            train_val_treatment.values, dtype=torch.float
        )
        test_treatment = torch.tensor(test_treatment.values, dtype=torch.float).to(
            device
        )
        train_val_outcome = torch.tensor(train_val_outcome.values, dtype=torch.float)
        test_outcome = torch.tensor(test_outcome.values, dtype=torch.float).to(device)

        logger.info("Running bootstrapping to find the confidence interval")
        try:
            hyperparams = hyperparams_df[
                (hyperparams_df["drug_0"] == combination[0])
                & (hyperparams_df["drug_1"] == combination[1])
            ]
            trained_lr = hyperparams["lr"].values[0]
            trained_hidden_dim = hyperparams["hidden_dim"].values[0]
            trained_n_in = hyperparams["n_in"].values[0]
            trained_n_out = hyperparams["n_out"].values[0]
        except Exception as e:
            hyperparams = hyperparams_df[
                (hyperparams_df["drug_1"] == combination[0])
                & (hyperparams_df["drug_0"] == combination[1])
            ]
            trained_lr = hyperparams["lr"].values[0]
            trained_hidden_dim = hyperparams["hidden_dim"].values[0]
            trained_n_in = hyperparams["n_in"].values[0]
            trained_n_out = hyperparams["n_out"].values[0]

        params = {
            "hidden_dim": trained_hidden_dim,
            "lr": trained_lr,
            "n_in": trained_n_in,
            "n_out": trained_n_out,
        }

        logger.info(f"Training the model with params: {params}")
        print(f"Training the model with params: {params}")

        ate_mean, ate_lower, ate_upper = bootstrap_ate_estimates(
            data,
            patient_severity,
            train_val_patient_ids,
            train_val_treatment,
            train_val_outcome,
            params,
        )

        variance = ate_upper - ate_lower

        bootstrap_df = create_bootstrap_df(
            ate_mean, ate_lower, ate_upper, variance, combination, "NetDeconf"
        )

        results_df = pd.concat([results_df, bootstrap_df], ignore_index=True)

    return results_df


def compute_network_deconfounder(
    final_data,
    ingredient_pairs,
    person_condition_network,
    person_drug_network,
    drug_condition_network,
    drug_drug_network,
    condition_condition_network,
    condition_concept_network,
):
    final_results_df = pd.DataFrame()
    tmp_num_people = 0
    for idx, combination in enumerate(ingredient_pairs):
        print(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )
        logger.info(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )

        filter_values_list = [int(x) for x in list(combination)]
        # ingredient_pairs = [(40234834, 710062)]
        # combination = (40234834, 710062)
        patient_severity = final_data.toPandas().copy()
        patient_severity["treatment"] = patient_severity["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        patient_severity = patient_severity[
            patient_severity["ingredient_concept_id"].isin(list(combination))
        ]
        patient_severity = patient_severity[
            ["person_id", "treatment", "severity_final"]
        ]
        patient_severity["person_id"] = patient_severity["person_id"].apply(
            lambda x: f"person_{x}"
        )

        person_condition = person_condition_network.alias("person_condition")
        person_drug = person_drug_network.alias("person_drug")
        drug_condition = drug_condition_network.alias("drug_condition")
        drug_drug = drug_drug_network.alias("drug_drug")
        condition_condition = condition_condition_network.alias("condition_condition")
        condition_concept = condition_concept_network.alias("condition_concept")

        person_condition = (
            person_condition.withColumn("condition", explode("condition_agg"))
            .withColumn("ingredient_concept_id", explode("ingredient_agg"))
            .select("person_id", "condition", "ingredient_concept_id")
            .dropDuplicates()
        )

        person_condition = (
            person_condition.withColumn(
                "node_1", concat(lit("person_"), col("person_id"))
            )
            .withColumn("node_2", concat(lit("condition_"), col("condition")))
            .filter(col("ingredient_concept_id").isin(filter_values_list))
            .withColumn("type", lit("person-condition"))
            .select("node_1", "node_2")
            .toPandas()
        )

        person_drug = (
            person_drug.withColumn("node_2", explode("node_2_agg"))
            .withColumn("ingredient_concept_id", explode("ingredient_agg"))
            .select("node_1", "node_2", "ingredient_concept_id")
        )

        person_drug = (
            person_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
            .select("node_1", "node_2")
            .toPandas()
        )

        drug_drug = (
            drug_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
            .select("node_1", "node_2")
            .toPandas()
        )

        condition_condition = condition_condition.select("node_1", "node_2").toPandas()

        drug_condition = (
            drug_condition.withColumn(
                "node_1", concat(lit("drug_"), col("drug_rxnorm_concept_id"))
            )
            .withColumn(
                "node_2", concat(lit("condition_"), col("condition_snomed_concept_id"))
            )
            .select("node_1", "node_2")
            .toPandas()
        )

        condition_concept = (
            condition_concept.withColumn(
                "node_1", concat(lit("condition_"), col("index_prefix"))
            )
            .withColumn("node_2", concat(lit("concept_"), col("concept_id")))
            .withColumn("type", lit("condition-condition_concept"))
            .select("node_1", "node_2")
            .toPandas()
        )

        # Compile all identifiers for each type into unified lists
        all_patient_ids = pd.concat(
            [person_drug["node_1"], person_condition["node_1"]]
        ).unique()

        patient_severity = patient_severity[
            patient_severity["person_id"].isin(all_patient_ids)
        ].reset_index(drop=True)

        all_drug_ids = pd.concat(
            [
                person_drug["node_2"],
                drug_drug["node_1"],
                drug_drug["node_2"],
                drug_condition["node_1"],
            ]
        ).unique()
        all_condition_ids = pd.concat(
            [
                person_condition["node_2"],
                condition_condition["node_1"],
                condition_condition["node_2"],
                drug_condition["node_2"],
                condition_concept["node_2"],
            ]
        ).unique()
        all_condition_classes = condition_concept["node_1"].unique()

        # Create global mappings with unique indices across all types
        global_index = 0
        patient_id_to_index = {}
        drug_id_to_index = {}
        condition_id_to_index = {}
        condition_class_to_index = {}
        
        # Assign indices for patients
        for patient in all_patient_ids:
            tmp_num_people += 1
            patient_id_to_index[patient] = global_index
            global_index += 1

        # Assign indices for drugs, offset by the number of patients
        for drug in all_drug_ids:
            drug_id_to_index[drug] = global_index
            global_index += 1

        for condition in all_condition_ids:
            condition_id_to_index[condition] = global_index
            global_index += 1

        for condition_class in all_condition_classes:
            condition_class_to_index[condition_class] = global_index
            global_index += 1

        # Apply mappings to data frames
        person_drug["node_1"] = person_drug["node_1"].map(patient_id_to_index)
        person_drug["node_2"] = person_drug["node_2"].map(drug_id_to_index)

        person_condition["node_1"] = person_condition["node_1"].map(patient_id_to_index)
        person_condition["node_2"] = person_condition["node_2"].map(
            condition_id_to_index
        )

        drug_drug["node_1"] = drug_drug["node_1"].map(drug_id_to_index)
        drug_drug["node_2"] = drug_drug["node_2"].map(drug_id_to_index)

        drug_condition["node_1"] = drug_condition["node_1"].map(drug_id_to_index)
        drug_condition["node_2"] = drug_condition["node_2"].map(condition_id_to_index)

        condition_condition["node_1"] = condition_condition["node_1"].map(
            condition_id_to_index
        )
        condition_condition["node_2"] = condition_condition["node_2"].map(
            condition_id_to_index
        )

        condition_concept["node_1"] = condition_concept["node_1"].map(
            condition_class_to_index
        )
        condition_concept["node_2"] = condition_concept["node_2"].map(
            condition_id_to_index
        )

        patient_severity["person_id"] = patient_severity["person_id"].map(
            patient_id_to_index
        )
        # patient_ids = patient_severity["person_id"]
        all_edges = pd.concat(
            [
                person_drug,
                person_condition,
                drug_drug,
                drug_condition,
                condition_condition,
                condition_concept,
            ]
        )

        # Get the (patient_id, treatment) pairs for the patients who have received both treatments
        selected_pairs = (
            patient_severity.groupby("person_id")
            .filter(lambda x: x["treatment"].nunique() > 1)
            .sort_values(by="person_id")
        )

        # From each unique (patient_id, treatment), randomly select one row
        selected_pairs = (
            selected_pairs.groupby("person_id")
            .apply(lambda x: x.sample(1))
            .reset_index(drop=True)
        )

        # Make a list of (patient_id, treatment) pairs
        pairs_to_drop = list(
            zip(selected_pairs["person_id"], selected_pairs["treatment"])
        )

        # Drop the selected pairs from the dataframe
        patient_severity = patient_severity[
            ~patient_severity.apply(
                lambda x: (x["person_id"], x["treatment"]) in pairs_to_drop, axis=1
            )
        ].reset_index(drop=True)

        # patient_ids = patient_severity['person_id']
        patient_severity = patient_severity.set_index("person_id")
        treatment = patient_severity["treatment"]
        outcome = patient_severity["severity_final"]

        edge_index = torch.tensor(all_edges.values, dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(edge_index.size(1))

        data = Data(
            num_nodes=global_index, edge_index=edge_index, edge_attr=edge_weight
        )

        test_roc, val_roc, y1, y0, best_ate, best_params, out_loss = net_deconf_main(
            data=data,
            # patient_node_indices=patient_ids,
            patient_severity=patient_severity,
            treatment=treatment,
            outcome=outcome,
        )

        best_params_df = create_best_params_df(
            best_params, test_roc, val_roc, y1, y0, best_ate, combination, "NetDeconf"
        )

        final_results_df = pd.concat(
            [final_results_df, best_params_df], ignore_index=True
        )

    return final_results_df, out_loss, tmp_num_people


# @configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
# @configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
# @transform(
#     output_df=Output("ri.foundry.main.dataset.019682ef-0b24-46a5-8cb2-31d3ff2205f6"),
#     concept_set_members=Input(
#         "ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"
#     ),
#     final_data=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
#     condition_mappings=Input(
#         "ri.foundry.main.dataset.b5c3702d-9383-4057-83ac-39a1844d007d"
#     ),
#     temp_person_condition_network=Input(
#         "ri.foundry.main.dataset.de088514-7d31-45ae-9898-35a88c46bc27"
#     ),
#     drug_condition=Input(
#         "ri.foundry.main.dataset.d6a29338-ca1a-4f13-85fd-7b82498484b4"
#     ),
#     person_drug=Input("ri.foundry.main.dataset.7b33023a-1c07-4ae6-88d3-e9d83de7ab7a"),
#     drug_drug=Input("ri.foundry.main.dataset.9eaab353-758b-4e66-a6a8-0ccb22935b0c"),
#     condition_condition=Input(
#         "ri.foundry.main.dataset.f3c801ad-fc9f-4356-95f0-491480362527"
#     ),
#     person_condition=Input(
#         "ri.foundry.main.dataset.de088514-7d31-45ae-9898-35a88c46bc27"
#     ),
#     condition_concept=Input(
#         "ri.foundry.main.dataset.0678aacb-6b1f-45ba-894e-5fe02a196f54"
#     ),
# )
# def compute(
#     output_df,
#     temp_person_condition_network,
#     concept_set_members,
#     final_data,
#     condition_mappings,
#     drug_condition,
#     person_drug,
#     drug_drug,
#     condition_condition,
#     person_condition,
#     condition_concept,
# ):
#     concept_set_members = concept_set_members.dataframe()
#     final_data = final_data.dataframe()
#     drug_condition_network = drug_condition.dataframe()
#     person_drug_network = person_drug.dataframe()
#     drug_drug_network = drug_drug.dataframe()
#     condition_condition_network = condition_condition.dataframe()
#     person_condition_network = person_condition.dataframe()
#     condition_concept_network = condition_concept.dataframe()

#     ingredient_list = final_data.toPandas().ingredient_concept_id.unique()
#     ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)

#     mini_batch_size = 50
#     mini_batches = []

#     for idx in range(0, len(ingredient_pairs), mini_batch_size):
#         mini_batches.append(ingredient_pairs[idx : idx + mini_batch_size])

#     final_results_df = pd.DataFrame()

#     for idx, combination in enumerate(mini_batches[0]):
#         print(
#             f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
#         )
#         logger.info(
#             f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
#         )

#         filter_values_list = [int(x) for x in list(combination)]
#         results_dict = {}
#         # ingredient_pairs = [(40234834, 710062)]
#         # combination = (40234834, 710062)
#         patient_severity = final_data.toPandas().copy()
#         patient_severity["treatment"] = patient_severity["ingredient_concept_id"].apply(
#             lambda x: 0 if x == combination[0] else 1
#         )
#         patient_severity = patient_severity[
#             patient_severity["ingredient_concept_id"].isin(list(combination))
#         ]
#         patient_severity = patient_severity[
#             ["person_id", "treatment", "severity_final"]
#         ]
#         patient_severity["person_id"] = patient_severity["person_id"].apply(
#             lambda x: f"person_{x}"
#         )

#         person_condition = person_condition_network.alias("person_condition")
#         person_drug = person_drug_network.alias("person_drug")
#         drug_condition = drug_condition_network.alias("drug_condition")
#         drug_drug = drug_drug_network.alias("drug_drug")
#         condition_condition = condition_condition_network.alias("condition_condition")
#         condition_concept = condition_concept_network.alias("condition_concept")

#         person_condition = (
#             person_condition.withColumn("condition", explode("condition_agg"))
#             .withColumn("ingredient_concept_id", explode("ingredient_agg"))
#             .select("person_id", "condition", "ingredient_concept_id")
#             .dropDuplicates()
#         )

#         person_condition = (
#             person_condition.withColumn(
#                 "node_1", concat(lit("person_"), col("person_id"))
#             )
#             .withColumn("node_2", concat(lit("condition_"), col("condition")))
#             .filter(col("ingredient_concept_id").isin(filter_values_list))
#             .withColumn("type", lit("person-condition"))
#             .select("node_1", "node_2")
#             .toPandas()
#         )

#         person_drug = (
#             person_drug.withColumn("node_2", explode("node_2_agg"))
#             .withColumn("ingredient_concept_id", explode("ingredient_agg"))
#             .select("node_1", "node_2", "ingredient_concept_id")
#         )

#         person_drug = (
#             person_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
#             .select("node_1", "node_2")
#             .toPandas()
#         )

#         drug_drug = (
#             drug_drug.filter(col("ingredient_concept_id").isin(filter_values_list))
#             .select("node_1", "node_2")
#             .toPandas()
#         )

#         condition_condition = condition_condition.select("node_1", "node_2").toPandas()

#         drug_condition = (
#             drug_condition.withColumn(
#                 "node_1", concat(lit("drug_"), col("drug_rxnorm_concept_id"))
#             )
#             .withColumn(
#                 "node_2", concat(lit("condition_"), col("condition_snomed_concept_id"))
#             )
#             .select("node_1", "node_2")
#             .toPandas()
#         )

#         condition_concept = (
#             condition_concept.withColumn(
#                 "node_1", concat(lit("condition_"), col("index_prefix"))
#             )
#             .withColumn("node_2", concat(lit("concept_"), col("concept_id")))
#             .withColumn("type", lit("condition-condition_concept"))
#             .select("node_1", "node_2")
#             .toPandas()
#         )

#         # Compile all identifiers for each type into unified lists
#         all_patient_ids = pd.concat(
#             [person_drug["node_1"], person_condition["node_1"]]
#         ).unique()

#         patient_severity = patient_severity[
#             patient_severity["person_id"].isin(all_patient_ids)
#         ].reset_index(drop=True)

#         all_drug_ids = pd.concat(
#             [
#                 person_drug["node_2"],
#                 drug_drug["node_1"],
#                 drug_drug["node_2"],
#                 drug_condition["node_1"],
#             ]
#         ).unique()
#         all_condition_ids = pd.concat(
#             [
#                 person_condition["node_2"],
#                 condition_condition["node_1"],
#                 condition_condition["node_2"],
#                 drug_condition["node_2"],
#                 condition_concept["node_2"],
#             ]
#         ).unique()
#         all_condition_classes = condition_concept["node_1"].unique()

#         # Create global mappings with unique indices across all types
#         global_index = 0
#         patient_id_to_index = {}
#         drug_id_to_index = {}
#         condition_id_to_index = {}
#         condition_class_to_index = {}

#         # Assign indices for patients
#         for patient in all_patient_ids:
#             patient_id_to_index[patient] = global_index
#             global_index += 1

#         # Assign indices for drugs, offset by the number of patients
#         for drug in all_drug_ids:
#             drug_id_to_index[drug] = global_index
#             global_index += 1

#         for condition in all_condition_ids:
#             condition_id_to_index[condition] = global_index
#             global_index += 1

#         for condition_class in all_condition_classes:
#             condition_class_to_index[condition_class] = global_index
#             global_index += 1

#         # Apply mappings to data frames
#         person_drug["node_1"] = person_drug["node_1"].map(patient_id_to_index)
#         person_drug["node_2"] = person_drug["node_2"].map(drug_id_to_index)

#         person_condition["node_1"] = person_condition["node_1"].map(patient_id_to_index)
#         person_condition["node_2"] = person_condition["node_2"].map(
#             condition_id_to_index
#         )

#         drug_drug["node_1"] = drug_drug["node_1"].map(drug_id_to_index)
#         drug_drug["node_2"] = drug_drug["node_2"].map(drug_id_to_index)

#         drug_condition["node_1"] = drug_condition["node_1"].map(drug_id_to_index)
#         drug_condition["node_2"] = drug_condition["node_2"].map(condition_id_to_index)

#         condition_condition["node_1"] = condition_condition["node_1"].map(
#             condition_id_to_index
#         )
#         condition_condition["node_2"] = condition_condition["node_2"].map(
#             condition_id_to_index
#         )

#         condition_concept["node_1"] = condition_concept["node_1"].map(
#             condition_class_to_index
#         )
#         condition_concept["node_2"] = condition_concept["node_2"].map(
#             condition_id_to_index
#         )

#         patient_severity["person_id"] = patient_severity["person_id"].map(
#             patient_id_to_index
#         )
#         patient_ids = patient_severity["person_id"]
#         all_edges = pd.concat(
#             [
#                 person_drug,
#                 person_condition,
#                 drug_drug,
#                 drug_condition,
#                 condition_condition,
#                 condition_concept,
#             ]
#         )

#         # Get the (patient_id, treatment) pairs for the patients who have received both treatments
#         selected_pairs = (
#             patient_severity.groupby("person_id")
#             .filter(lambda x: x["treatment"].nunique() > 1)
#             .sort_values(by="person_id")
#         )

#         # From each unique (patient_id, treatment), randomly select one row
#         selected_pairs = (
#             selected_pairs.groupby("person_id")
#             .apply(lambda x: x.sample(1))
#             .reset_index(drop=True)
#         )

#         # Make a list of (patient_id, treatment) pairs
#         pairs_to_drop = list(
#             zip(selected_pairs["person_id"], selected_pairs["treatment"])
#         )

#         # Drop the selected pairs from the dataframe
#         patient_severity = patient_severity[
#             ~patient_severity.apply(
#                 lambda x: (x["person_id"], x["treatment"]) in pairs_to_drop, axis=1
#             )
#         ].reset_index(drop=True)

#         patient_severity = patient_severity.set_index("person_id")
#         treatment = patient_severity["treatment"]
#         outcome = patient_severity["severity_final"]

#         print(f"Patient_ids shape: {patient_ids.shape}\n")
#         logger.info(f"Patient_ids shape: {patient_ids.shape}\n")

#         edge_index = torch.tensor(all_edges.values, dtype=torch.long).t().contiguous()
#         edge_weight = torch.ones(edge_index.size(1))

#         data = Data(
#             num_nodes=global_index, edge_index=edge_index, edge_attr=edge_weight
#         )

#         print(f"Number of nodes: {data.num_nodes}")
#         print(f"Number of edges: {data.num_edges}")

#         logger.info(f"Number of nodes: {data.num_nodes}")
#         logger.info(f"Number of edges: {data.num_edges}")
#         logger.info(data)

#         ate, acc, roc = net_deconf_main(
#             data=data,
#             patient_node_indices=patient_ids,
#             patient_severity=patient_severity,
#             treatment=treatment,
#             outcome=outcome,
#         )

#         results_dict["drug_0"] = combination[0]
#         results_dict["drug_1"] = combination[1]
#         results_dict["acc"] = acc
#         results_dict["roc"] = roc
#         results_dict["ate"] = ate

#         # results_df = pd.DataFrame([results_dict])
#         final_results_df = pd.concat(
#             [final_results_df, pd.DataFrame([results_dict])], ignore_index=True
#         )

#     # Create a SparkSession object
#     spark = SparkSession.builder.getOrCreate()
#     logger.info("No issues till now")
#     # print("No issues till now")

#     # Convert the Pandas DataFrame to a PySpark DataFrame
#     df_pyspark = spark.createDataFrame(final_results_df)
#     logger.info("No issues till now")
#     # print(f"No issues till now. Type of df {type(df_pyspark)}")

#     return output_df.write_dataframe(df_pyspark)
