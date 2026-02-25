from pyspark.sql import functions as F
from transforms.api import transform, Input, Output

# Sklearn packages
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from causalml.inference.nn import CEVAE

import numpy as np
import pandas as pd
from itertools import combinations


def metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts):
    yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(
        list(yhat_ts.values())[0]
    )
    preds = (1.0 - t_valid) * yhat_cs + t_valid * yhat_ts
    roc = roc_auc_score(y_valid, preds)
    ate = ite.mean()
    return roc, ate


def create_best_params_df(best_params, best_roc, best_ate, combination, model):
    best_params["roc"] = best_roc
    best_params["ate"] = best_ate
    best_params["drug_0"] = combination[0]
    best_params["drug_1"] = combination[1]
    best_params["model"] = model
    return pd.DataFrame(best_params, index=[0])


def grid_search(
    X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict
):
    best_roc = 0.0
    best_ate = 0.0
    estim_1 = [100, 200]
    estim_2 = ["sqrt", "log2", None]
    estim_3 = [7, 10]
    for crit_1 in estim_1:
        for crit_2 in estim_2:
            for crit_3 in estim_3:
                clf = RandomForestClassifier(
                    n_estimators=crit_1,
                    max_features=crit_2,
                    max_depth=crit_3,
                    class_weight=class_weight_dict,
                )
                clf_learner = BaseSClassifier(learner=clf)
                clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                ite, yhat_cs, yhat_ts = clf_learner.predict(
                    X=X_valid,
                    treatment=t_valid,
                    y=y_valid,
                    return_components=True,
                    verbose=True,
                )
                roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts)
                if roc > best_roc:
                    best_ate = ate
                    best_roc = roc
                    best_params = {
                        "n_estimators": str(crit_1),
                        "criterion": str(crit_2),
                        "max_depth": str(crit_3),
                        "penalty": str(None),
                        "C": str(None),
                        "max_iter": str(None),
                        "solver": str(None),
                    }
    return best_roc, best_ate, best_params


def main_calculation(main_df, ingredient_pairs):
    results_df = pd.DataFrame()
    for idx, combination in enumerate(ingredient_pairs):
        # start_time = datetime.now()
        # print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.where(F.col("ingredient_concept_id").isin(list(combination)))
        df = df.withColumn(
            "treatment",
            F.when(
                F.col("ingredient_concept_id") == combination[0], F.lit(1)
            ).otherwise(
                F.when(F.col("ingredient_concept_id") == combination[1], F.lit(0))
            ),
        )
        df = df.toPandas()
        print(df.shape)  # noqa

        # Inputs: covariates and treatment
        X = df.drop(
            ["person_id", "severity_final", "ingredient_concept_id", "treatment"],
            axis=1,
        )
        y = df["severity_final"]
        t = df["treatment"]

        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=2, stratify=y
        )
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test, y_test, test_size=0.5, random_state=2, stratify=y_test
        )
        y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        t_train = t[X_train.index]
        t_train = t_train.values
        t_test = t[X_test.index]
        t_test = t_test.values
        t_valid = t[X_valid.index]
        t_valid = t_valid.values

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params = grid_search(
            X_train,
            y_train,
            t_train,
            X_valid,
            y_valid,
            t_valid,
            class_weight_dict,
        )
        print(f"ROC: {best_roc}, {best_params}")
        best_params_df = create_best_params_df(
            best_params, best_roc, best_ate, combination, "RF"
        )
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        return results_df


@transform(
    output_df=Output("ri.foundry.main.dataset.26eeecab-183a-422d-a36e-34345cdb4858"),
    source_df=Input("ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498"),
)
def compute(output_df, source_df):
    main_df = source_df.dataframe()
    # ingredient_list = main_df.select(F.collect_set('ingredient_concept_id').alias('ingredient_concept_id')).first()['ingredient_concept_id']
    # ingredient_pairs = list(combinations(ingredient_list, 2))
    # ingredient_pairs = [(739138, 703547)]
    ingredient_pairs = [(778268, 751412)]

    results_df = main_calculation(main_df, ingredient_pairs)
    # print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    # print('Total time taken:',datetime.now() - initial_time)
    return output_df.write_dataframe(results_df)
