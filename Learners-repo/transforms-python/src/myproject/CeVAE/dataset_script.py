from sklearn import model_selection
import pandas as pd
import numpy as np


class CustomDataset(object):
    def __init__(self, main_df, combination):
        # Initialize instance variables
        self.main_df = main_df
        self.combination = combination
        self.results_df = pd.DataFrame()
        self.binfeats = [
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            # 32,
        ]  # Binary features
        self.confeats = [1]  # Continous features

    def get_train_valid_test(self):
        df = self.main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(self.combination))].reset_index(
            drop=True
        )
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == self.combination[0] else 1
        )

        # Set mu0 and mu1
        df["mu0"] = df[df["treatment"] == 0]["severity_final"].mean()
        df["mu1"] = df[df["treatment"] == 1]["severity_final"].mean()

        mu0 = df["mu0"].reset_index(drop=True)
        mu1 = df["mu1"].reset_index(drop=True)

        X = df.drop(
            [
                "person_id",
                "severity_final",
                "ingredient_concept_id",
                "treatment",
                "mu0",
                "mu1",
            ],
            axis=1,
        )
        X["age_at_covid"] = (X["age_at_covid"] - X["age_at_covid"].min()) / (
            X["age_at_covid"].max() - X["age_at_covid"].min()
        )
        X = X.reset_index(drop=True)
        y = df["severity_final"]
        T = df["treatment"]

        np.random.seed(42)
        idxtrain, ite = model_selection.train_test_split(
            np.arange(X.shape[0]), test_size=0.1, random_state=1
        )
        itr, iva = model_selection.train_test_split(
            idxtrain, test_size=0.3, random_state=1
        )
        train = (X.iloc[itr], T.iloc[itr], y.iloc[itr]), (mu0.iloc[itr], mu1.iloc[itr])
        valid = (X.iloc[iva], T.iloc[iva], y.iloc[iva]), (mu0.iloc[iva], mu1.iloc[iva])
        test = (X.iloc[ite], T.iloc[ite], y.iloc[ite]), (mu0.iloc[ite], mu1.iloc[ite])
        return train, valid, test, self.binfeats, self.confeats
