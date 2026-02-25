# from pyspark.sql import functions as F
from transforms.api import transform, Input, Output, configure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split
)
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from itertools import combinations
import logging
@configure(["DRIVER_GPU_ENABLED", "DRIVER_MEMORY_EXTRA_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_GPU_ENABLED", "EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_LARGE"])
@transform(
    output = Output("ri.foundry.main.dataset.f0d86173-8201-477a-9b3c-7401454c9b57"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def compute(source_df, output):
    source_df = source_df.dataframe()
    source_df = source_df.toPandas()
    combination = (717607, 725131)
    df = source_df.copy()
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

    # Fit logistic regression with L1 penalty
    lasso_logreg = LogisticRegression(penalty='l1', solver='liblinear')
    lasso_logreg.fit(X_train, y_train)

    # Create DataFrame of feature names and coefficients
    coef_df = pd.DataFrame(zip(X.columns, lasso_logreg.coef_[0]), columns=['feature', 'coefficient'])

    # Convert all columns to string type
    coef_df = coef_df.astype(str)

    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Convert to PySpark DataFrame
    df_pyspark = spark.createDataFrame(coef_df)

    # Output the PySpark DataFrame
    return output.write_dataframe(df_pyspark)

