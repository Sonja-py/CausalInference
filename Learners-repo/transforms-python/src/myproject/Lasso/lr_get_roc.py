# from pyspark.sql import functions as F
from transforms.api import transform, Input, Output, configure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV
)
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from itertools import combinations
import logging
from sklearn.metrics import roc_auc_score

@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output = Output("ri.foundry.main.dataset.f8d55382-82c7-41c9-8135-27925621f9c1"),
    source_df = Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)

def compute(source_df, output):
    # Load and prep data
    df = source_df.dataframe().toPandas()
    # combination = (717607, 725131)
    top10 = [(717607, 725131), (750982, 725131), (44507700, 725131),
        (715939, 725131), (755695, 725131), (778268, 722031), (797617, 725131),
        (739138, 725131), (717607, 715259), (738156, 725131)]
    top10 = [(717607, 725131)]
    features = [
        "SICKLECELLDISEASE_before_or_day_of_covid_indicator",
        "DEMENTIA_before_or_day_of_covid_indicator",
        "METASTATICSOLIDTUMORCANCERS_before_or_day_of_covid_indicator",
        "PSYCHOSIS_before_or_day_of_covid_indicator",
        "HEMIPLEGIAORPARAPLEGIA_before_or_day_of_covid_indicator",
    ]
    param_grid = [{'penalty':['l1','l2'],
    'C' : np.logspace(-3,3,10),
    'solver': ['liblinear'],
    'max_iter': [1000, 5000,10000]
    }]
    out=pd.DataFrame()

    for combination in  top10:
        df_curr = df[df.ingredient_concept_id.isin(list(combination))]
        df_curr["treatment"] = df_curr["ingredient_concept_id"].apply(
                lambda x: 0 if x == combination[0] else 1
            )
    
        X = df_curr[features]
        y = df_curr["severity_final"]
        t = df_curr["treatment"]

        # Train/test split
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced")
        grid = GridSearchCV(model, param_grid, cv=5)

        grid.fit(X_train, y_train)
        pred = grid.predict_proba(X_test)[:, 1]

        # Compute factual ROC AUC
        roc_auc = roc_auc_score(y_test, pred)

        # Save to DataFrame
        result_df = pd.DataFrame([{"factual_roc_auc": roc_auc, "drug0": combination[0], "drug1": combination[1]}])
        out = pd.concat([result_df, out], ignore_index=True)

    # Convert to PySpark
    spark = SparkSession.builder.getOrCreate()
    df_pyspark = spark.createDataFrame(out)

    return output.write_dataframe(df_pyspark)

