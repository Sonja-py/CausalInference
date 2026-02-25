from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure
from itertools import combinations
import logging

from myproject.CeVAE.cevae_main_no_fold import cevae_func
from myproject.CeVAE.pyro_cevae import CEVAE
import pyro

import numpy as np
import pandas as pd
import torch
import torch.distributions
import random 

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output_df=Output("ri.foundry.main.dataset.4e419b82-70a8-4f99-955a-cb8f0890d269"),
    source_df = Input("ri.foundry.main.dataset.58516625-13eb-4b88-90f1-21cecc1ce7bb"),
)
def compute(output_df, source_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    df = source_df.dataframe().toPandas()
    combination = (750982, 725131)
    
    features = [
        "SICKLECELLDISEASE_before_or_day_of_covid_indicator",
        "DEMENTIA_before_or_day_of_covid_indicator",
        "METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator",
        "PSYCHOSIS_before_or_day_of_covid_indicator",
        "HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator",
    ]
    df = df[df.ingredient_concept_id.isin(list(combination))]
    df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
   
    X = df[features]
    y = df["severity_final"]
    t = df["treatment"]


    # Train/test split
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42, stratify=y
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    X_train = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)
    t_train = torch.tensor(t_train.values, dtype=torch.float32, device=device)
    t_test = torch.tensor(t_test.values, dtype=torch.float32, device=device)
    
    param_grid = {
        "num_layers": [3, 4],
        "hidden_dim": [64, 128],
        "learning_rate": [1e-4, 5e-3],
        "num_epochs": [100, 200],
        "latent_dim": [20],
        "num_samples": [100],
        "batch_size": [200]
    }
    grid = ParameterGrid(param_grid)
    roc_count = 20
    roc_total = 0
    best_roc = -np.inf
    for param in grid:
        cevae = CEVAE(
                feature_dim=X_train.size(1), 
                num_layers=param["num_layers"],
                hidden_dim=param["hidden_dim"],
                num_samples=param["num_samples"])
        
        losses = cevae.fit(
                X_train, 
                t_train, 
                y_train, 
                num_epochs=param["num_epochs"],
                learning_rate=param["learning_rate"],
                batch_size=param["batch_size"]
                )

        for i in range(roc_count):
            random_seed = random.randint(0, 100)
            pyro.set_rng_seed(random_seed)
            ites, y0, y1 = cevae.ite(X_test)
            ate = torch.mean(torch.flatten(ites))
            
            # flatten ys
            y0 = y0.mean(dim=0).flatten()
            y1 = y1.mean(dim=0).flatten()

            # roc
            # y0n = (torch.sigmoid(y0)>0.5).float()
            # y1n = (torch.sigmoid(y1)>0.5).float()
            y = torch.where(t_test > 0, y1, y0)

            roc = roc_auc_score(y_test.cpu(), y.cpu())
            roc_total += roc
        ROC = roc_total/roc_count
        roc_total = 0
        if ROC > best_roc:
            best_roc = ROC


   # Save to DataFrame
    result_df = pd.DataFrame([{"factual_roc_auc": best_roc}])

    # Convert to PySpark
    spark = SparkSession.builder.getOrCreate()
    df_pyspark = spark.createDataFrame(result_df)

    return output_df.write_dataframe(df_pyspark)

