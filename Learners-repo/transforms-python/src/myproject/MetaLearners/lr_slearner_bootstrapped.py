from transforms.api import transform, Input, Output, configure
import pickle
import pandas as pd
from pyspark.sql import SparkSession

from itertools import combinations
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

logger = logging.getLogger()


def sample(file_df, filename):
    df = file_df
    fs = df.filesystem()  # This is the FileSystem object.
    try:
        with fs.open(f"{filename[0]}.pickle", mode="rb") as f:
            # print(f'{filename[0]}.pickle Found')
            model = pickle.load(f)
            # print(model)
            return model
    except Exception as e:
        # print(f'The error is: {e}')
        with fs.open(f"{filename[1]}.pickle", mode="rb") as f:
            # print(f'{filename[1]}.pickle Found')
            model = pickle.load(f)
            # print(model)
            return model
    return "if statement not working"


def create_best_params_df(ate, ate_lower, ate_upper, variance, combination, model):
    best_params = {}
    best_params["ate"] = ate
    best_params["ate_lower"] = ate_lower
    best_params["ate_upper"] = ate_upper
    best_params["variance"] = variance
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


def temp(X_test, y_test, t_test, class_weight_dict, clf_learner):
    ate, ate_lower, ate_upper = clf_learner.estimate_ate(
        X=X_test,
        treatment=t_test,
        y=y_test,
        return_ci=True,
        bootstrap_ci=True,
        n_bootstraps=100,
        bootstrap_size=10000,
        # pretrain=True,
    )

    logger.info(f"ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}")
    return ate[0], ate_lower[0], ate_upper[0]


def main_func(final_data, lr_slearner_df):
    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(
        columns=[
            "ate",
            "ate_lower",
            "ate_upper",
            "variance",
            "drug_0",
            "drug_1",
            "model",
        ]
    )
    # ingredient_list = main_df.ingredient_concept_id.unique()
    # ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    ingredient_pairs = [(40234834, 710062)]

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        logger.info(
            f"-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------"
        )
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )

        X = df.drop(
            ["person_id", "severity_final", "ingredient_concept_id", "treatment"],
            axis=1,
        )
        y = df["severity_final"]
        t = df["treatment"]

        np.random.seed(0)
        (
            X_train_val,
            X_test,
            y_train_val,
            y_test,
            t_train_val,
            t_test,
        ) = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(enumerate(class_weights))

        clf_learner = sample(
            lr_slearner_df,
            [
                f"{combination[0]}_{combination[1]}",
                f"{combination[1]}_{combination[0]}",
            ],
        )
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, class_weight_dict, clf_learner)

        # results_df.loc[-1] = [ate, ate_l, ate_u, ate_u - ate_l, combination[0], combination[1], 'S_LR']
        params_df = create_best_params_df(
            ate, ate_l, ate_u, ate_u - ate_l, combination, "S_LR"
        )
        results_df = pd.concat([results_df, params_df], ignore_index=True)

        logger.info(
            f"Time taken for combination {idx+1} is {datetime.now() - start_time}"
        )

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    logger.info("Total time taken:", datetime.now() - initial_time)
    return results_df


@configure(["DRIVER_MEMORY_LARGE", "DRIVER_CORES_LARGE"])
@configure(["EXECUTOR_MEMORY_MEDIUM", "EXECUTOR_CORES_SMALL"])
@transform(
    lr_slearner_df=Input(
        "ri.foundry.main.dataset.67236741-6d93-418d-83c3-91a2b3ea8405"
    ),
    final_data=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
    processed=Output("ri.foundry.main.dataset.3e63b3d1-eb32-4d11-9943-8b137d1cd249"),
)
def compute_df(lr_slearner_df, final_data, processed):
    # lr_slearner_df = lr_slearner_df.dataframe()
    final_data = final_data.dataframe()

    results_df = main_func(final_data, lr_slearner_df)

    # Create a SparkSession object
    spark = SparkSession.builder.getOrCreate()
    logger.info("No issues till now")
    # print("No issues till now")

    # Convert the Pandas DataFrame to a PySpark DataFrame
    df_pyspark = spark.createDataFrame(results_df)
    logger.info("No issues till now")
    # print(f"No issues till now. Type of df {type(df_pyspark)}")

    return processed.write_dataframe(df_pyspark)
