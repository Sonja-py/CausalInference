from pyspark.sql import SparkSession
from transforms.api import transform, Input, Output, configure
import pandas as pd
from itertools import combinations
import logging
from myproject.CeVAE.cevae_main import cevae_func
from imblearn.over_sampling import RandomOverSampler
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

logger = logging.getLogger()


@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output_25=Output("ri.foundry.main.dataset.e9bfc4a2-4919-4b4b-9f30-edc760fc8acc"),
    output_50=Output("ri.foundry.main.dataset.7e056689-d3f3-477a-99ed-2829628ea25f"),
    output_75=Output("ri.foundry.main.dataset.8181188a-2fc9-4987-9097-2ffc58a0eba7"),
    output_df=Output("ri.foundry.main.dataset.3c73987b-4bdd-4bac-92a4-4e0a89081f06"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)
def compute(output_25, output_50, output_75, output_df, source_df):
    top10 = [(717607, 725131), (750982, 725131), (44507700, 725131),
    (715939, 725131), (755695, 725131), (778268, 722031), (797617, 725131),
    (739138, 725131), (717607, 715259), (738156, 725131)]

    source_df = source_df.dataframe()
    source_df = source_df.toPandas()

    results_df = pd.DataFrame()
    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)
    mini_batches = []
    
    for i, pair in enumerate(ingredient_pairs):
        if pair in top10[8:9]:
            mini_batches.append(pair)
    # try:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    results_df = pd.DataFrame()
    param_grid = {
        "num_layers": [3, 4],
        "hidden_dim": [64, 128],
        "learning_rate": [5e-3, 1e-4],
        "num_epochs": [100, 200],
        "latent_dim": [20],
        "num_samples": [100],
        "batch_size": [200]
    }
    grid = ParameterGrid(param_grid)
    features = [
        "SICKLECELLDISEASE_before_or_day_of_covid_indicator",
        "DEMENTIA_before_or_day_of_covid_indicator",
        "METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator",
        "PSYCHOSIS_before_or_day_of_covid_indicator",
        "HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator",
        "treatment"
    ]
    spark = SparkSession.builder.getOrCreate()
    for idx, combination in enumerate(mini_batches):
        df = source_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        X = df[features]
        y = df["severity_final"]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
 
        treatment_series = X_train_val["treatment"]
        X_train_val_without_treatment = X_train_val.drop(columns=["treatment"], axis=1)
        
        total = len(grid)
        spark = SparkSession.builder.getOrCreate()

        for i, param in enumerate(grid):
            results_df = cevae_func(
                param,
                X_train_val_without_treatment,
                y_train_val,
                treatment_series,
                combination,
                results_df
            )

            progress_ratio = (i + 1) / total
            if abs(progress_ratio - 0.25) < 1e-6 or abs(progress_ratio - 0.5) < 1e-6 or abs(progress_ratio - 0.75) < 1e-6 or i == total - 1:
                df_pyspark = spark.createDataFrame(results_df)
                if abs(progress_ratio - 0.25) < 1e-6:
                    output_25.write_dataframe(df_pyspark)
                elif abs(progress_ratio - 0.5) < 1e-6:
                    output_50.write_dataframe(df_pyspark)
                elif abs(progress_ratio - 0.75) < 1e-6:
                    output_75.write_dataframe(df_pyspark)
                elif i == total - 1:
                    return output_df.write_dataframe(df_pyspark)
        return output_df.write_dataframe(df_pyspark)
