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

def create_ate_df(ate, roc, combination, model, parameters, best_val_roc):
    things = {}
    things["ATE"] = ate
    things["ROC"] = roc
    things["val_ROC"] = best_val_roc
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


def cevae_func(main_df, ingredient_pairs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # torch.set_default_device("cuda")
    """
    Implement CEVAE method

    Args:
        main_df: patient records for a given treatment-control group
        ingredient_pairs: treatments
    """

    # param_grid = {
    #     "num_layers": [3,4],
    #     "hidden_dim": [256, 512],
    #     "learning_rate": [5e-4, 1e-4, 5e-5, 1e-5],
    #     "num_epochs": [200],
    #     "latent_dim": [20, 30],
    #     "num_samples": [100, 150],
    #     "batch_size": [200, 300]
    # }
    param_grid = {
        "num_layers": [3],
        "hidden_dim": [256],
        "learning_rate": [1e-4],
        "num_epochs": [200],
        "latent_dim": [20],
        "num_samples": [500],
        "batch_size": [200]
    }
    grid = ParameterGrid(param_grid)

    results_df = pd.DataFrame()
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    count = 30
    outs = []
    
    for idx, combination in enumerate(ingredient_pairs):
        best_ROC = -np.inf
        best_params = None
        best_model = None
        
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df["treatment"] = df["ingredient_concept_id"].apply(
            lambda x: 0 if x == combination[0] else 1
        )
        X = df.drop(
            ["severity_final", "ingredient_concept_id"],
            axis=1
        )
        y = df["severity_final"]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        # people = X_test["person_id"]
        X = X.drop(["person_id"], axis=1)
        X_train_val = X_train_val.drop(["person_id"], axis=1)
        X_test = X_test.drop(["person_id"], axis=1)
        treatment_series = X_train_val["treatment"]
        X_train_val_without_treatment = X_train_val.drop(columns=["treatment"], axis=1)
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X_train_val, y_train_val)
    
    
        t = X_test["treatment"]
        X_test_without_treatment = X_test.drop(columns=["treatment"])
        X_test = torch.tensor(X_test_without_treatment.to_numpy(), dtype=torch.float32)
        t_test = torch.tensor(t.to_numpy(), dtype=torch.float32)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)  # Drop 'treatment' from features
        
        X_train_val_without_treatment = torch.tensor(X_train_val_without_treatment.to_numpy(), dtype=torch.float32).to(device)
        y_train_val = torch.tensor(y_train_val.to_numpy(), dtype=torch.float32).to(device)
        treatment_series = torch.tensor(treatment_series.to_numpy(), dtype=torch.float32).to(device)

        for param in grid:
#                pyro.set_rng_seed(0)
#                cevae = CEVAE(
#                        feature_dim=X_train_val_without_treatment.size(1), 
#                        num_layers=param["num_layers"],
#                        hidden_dim=param["hidden_dim"],
#                        num_samples=param["num_samples"])
#
#                losses = cevae.fit(
#                        X_train_val_without_treatment, 
#                        treatment_series, 
#                        y_train_val, 
#                        num_epochs=param["num_epochs"],
#                        learning_rate=param["learning_rate"],
#                        batch_size=param["batch_size"]
                    #)

            #random_seed = random.randint(0, 100)
            #pyro.set_rng_seed(random_seed)
            #ites, y0, y1 = cevae.ite(X_test)
            #ate = torch.mean(torch.flatten(ites))
            
            # flatten ys
            #y0 = y0.mean(dim=0).flatten()
            #y1 = y1.mean(dim=0).flatten()
            
            # roc
            # y0n = (torch.sigmoid(y0)>0.5).float()
            # y1n = (torch.sigmoid(y1)>0.5).float()
            #y = torch.where(t_test > 0, y1, y0)
            #outs.append(y)
            #ROC = roc_auc_score(y_test.cpu(), y.cpu())

            #if ROC > best_ROC:
            best_params = param
                #    # best_loss = mean_loss
                #    best_model = cevae
                #    best_val_roc = ROC
                #    best_ate = ate
    # matrix = np.zeros((count, count))
    #for i, o1 in enumerate(outs):
    #    for j, o2 in enumerate(outs):
    #        corr, x = spearmanr(outs[i].cpu(), outs[j].cpu())
    #        matrix[i][j] = corr
    #matrix = np.round(matrix, 3)
    #avg = (np.sum(matrix) - count) / (count*(count-1))
    #return pd.DataFrame({"corr": [str(avg)]})
    #        t = X_test["treatment"]
     #       X_test_without_treatment = X_test.drop(columns=["treatment"])
      #      X_test = torch.tensor(X_test_without_treatment.to_numpy(), dtype=torch.float32)
       #     t_test = torch.tensor(t.to_numpy(), dtype=torch.float32)
        #    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)  # Drop 'treatment' from features
            
#            X_train_val_without_treatment = torch.tensor(X_train_val_without_treatment.to_numpy(), dtype=torch.float32).to(device)
#            y_train_val = torch.tensor(y_train_val.to_numpy(), dtype=torch.float32).to(device)
#            treatment_series = torch.tensor(treatment_series.to_numpy(), dtype=torch.float32).to(device)
        cevae = CEVAE(
                feature_dim=X_train_val_without_treatment.size(1), 
                num_layers=best_params["num_layers"],
                hidden_dim=best_params["hidden_dim"],
                num_samples=best_params["num_samples"])
        
        for i in range(count):
            losses = cevae.fit(
                    X_train_val_without_treatment, 
                    treatment_series, 
                    y_train_val, 
                    num_epochs=param["num_epochs"],
                    learning_rate=param["learning_rate"],
                    batch_size=param["batch_size"]
                    )


            ites, y0, y1 = cevae.ite(X_test)
            ate = torch.mean(torch.flatten(ites))
            
            # flatten ys
            y0 = y0.mean(dim=0).flatten()
            y1 = y1.mean(dim=0).flatten()

            # roc
            # y0n = (torch.sigmoid(y0)>0.5).float()
            # y1n = (torch.sigmoid(y1)>0.5).float()
            y = torch.where(t_test > 0, y1, y0)

            ROC = roc_auc_score(y_test.cpu(), y.cpu())

            out_df = create_ate_df(ate.item(), ROC, combination, "CEVAE", best_params, ROC)
            results_df = pd.concat([results_df, out_df], ignore_index=True)

    return pd.DataFrame(results_df)

# from importlib import reload
# import logging

# import numpy as np
# import pandas as pd
# import torch
# import torch.distributions
# from sklearn.model_selection import ParameterGrid
# from sklearn import metrics
# # from pyro.contrib.cevae import CEVAE
# from myproject.CeVAE.pyro_cevae import CEVAE
# from sklearn.metrics import roc_auc_score

# from sklearn.model_selection import (
#     train_test_split,
#     StratifiedKFold
# )

# def create_ate_df(ate, roc, combination, model, parameters, best_val_roc):
#     things = {}
#     things["ATE"] = ate
#     things["ROC"] = roc
#     things["val_ROC"] = best_val_roc
#     things["num_layers"] = parameters["num_layers"]
#     things["hidden_dim"] = parameters["hidden_dim"]
#     things["learning_rate"] = parameters["learning_rate"]
#     things["num_epochs"] = parameters["num_epochs"]
#     things["num_samples"] = parameters["num_samples"]
#     things["drug_0"] = combination[0]
#     things["drug_1"] = combination[1]
#     things["model"] = model
#     return pd.DataFrame(things, index=[0])


# def cevae_func(main_df, ingredient_pairs):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)

#     # torch.set_default_device("cuda")
#     """
#     Implement CEVAE method

#     Args:
#         main_df: patient records for a given treatment-control group
#         ingredient_pairs: treatments
#     """

#     param_grid = {
#         "num_layers": [3],
#         "hidden_dim": [256],
#         "learning_rate": [1e-4],
#         "num_epochs":  [200],
#         "num_samples":  [100], 
#     }
#     grid = ParameterGrid(param_grid)

#     results_df = pd.DataFrame()
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     for idx, combination in enumerate(ingredient_pairs):
#         best_ROC = -np.inf
#         best_params = None
#         best_model = None
        
#         df = main_df.copy()
#         df = df[df.ingredient_concept_id.isin(list(combination))]
#         df["treatment"] = df["ingredient_concept_id"].apply(
#             lambda x: 0 if x == combination[0] else 1
#         )
#         X = df.drop(
#             ["severity_final", "ingredient_concept_id"],
#             axis=1
#         )
#         y = df["severity_final"]
        
#         X_train_val, X_test, y_train_val, y_test = train_test_split(
#             X, 
#             y, 
#             test_size=0.2, 
#             random_state=42, 
#             stratify=y
#         )
 
#         # people = X_test["person_id"]
#         X = X.drop(["person_id"], axis=1)
#         X_train_val = X_train_val.drop(["person_id"], axis=1)
#         X_test = X_test.drop(["person_id"], axis=1)
#         treatment_series = X_train_val["treatment"]
#         X_train_val_without_treatment = X_train_val.drop(columns=["treatment"], axis=1)
#         for param in grid:
#             # logger.info(f"Training with params: {param}")
#             # print(f"Training with params: {param}")

#             fold_scores = []
#             for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_without_treatment, y_train_val)):
#                 t_train_fold, t_val_fold = treatment_series.iloc[train_idx], treatment_series.iloc[val_idx]
#                 # Create the training and validation sets for each fold
#                 X_train_fold, X_val_fold = X_train_val_without_treatment.iloc[train_idx], X_train_val_without_treatment.iloc[val_idx]
#                 y_train_fold, y_val_fold = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
#                 X_train_fold = torch.tensor(X_train_fold.to_numpy(), dtype=torch.float32).to(device)
#                 X_val_fold = torch.tensor(X_val_fold.to_numpy(), dtype=torch.float32).to(device)
#                 t_train_fold = torch.tensor(t_train_fold.to_numpy(), dtype=torch.float32).to(device)
#                 t_val_fold = torch.tensor(t_val_fold.to_numpy(), dtype=torch.float32).to(device)
#                 y_train_fold = torch.tensor(y_train_fold.to_numpy(), dtype=torch.float32).to(device)
#                 y_val_fold = torch.tensor(y_val_fold.to_numpy(), dtype=torch.float32).to(device)
                
#                 cevae = CEVAE(
#                     feature_dim=X_train_fold.size(1), 
#                     num_layers=param["num_layers"],
#                     hidden_dim=param["hidden_dim"],
#                     num_samples=param["num_samples"]).to(device)
#                 # devices_info = {
#                 #     'X_train_fold': [str(X_train_fold.device)],
#                 #     'X_val_fold': [str(X_val_fold.device)],
#                 #     't_train_fold': [str(t_train_fold.device)],
#                 #     't_val_fold': [str(t_val_fold.device)],
#                 #     'y_train_fold': [str(y_train_fold.device)],
#                 #     'y_val_fold': [str(y_val_fold.device)],
#                 # }

#                 # # Convert to dataframe
#                 # devices_df = pd.DataFrame(devices_info)

#                 # return devices_df
#                 losses = cevae.fit(
#                     X_train_fold, 
#                     t_train_fold, 
#                     y_train_fold, 
#                     num_epochs=param["num_epochs"],
#                     learning_rate=param["learning_rate"],
#                     batch_size=200)
#                 mean_loss = np.array(losses).mean()
#                 # return pd.DataFrame(np.array(losses))

#                 val_ites, val_y0, val_y1 = cevae.ite(X_val_fold)
#                 ate = torch.mean(torch.flatten(val_ites))

#                 # flatten ys
#                 y0 = val_y0.mean(dim=0).flatten()
#                 y1 = val_y1.mean(dim=0).flatten()
#                 y = torch.where(t_val_fold > 0, y1, y0)

#                 df_results = pd.DataFrame({
#                     "treatment": t_val_fold.cpu(),
#                     "true_y": y_val_fold.cpu().numpy(),
#                     "predicted_y": y.cpu(),
#                     "y1": y1.cpu(),
#                     "y0": y0.cpu(),
#                 })
                
#                 return df_results
#                 ROC = roc_auc_score(y_val_fold.cpu(), y.cpu())
#                 fold_scores.append(ROC)

#             mean_scores = np.mean(fold_scores)
#             if mean_scores > best_ROC:
#                 best_params = param
#                 # best_loss = mean_loss
#                 best_model = cevae
#                 best_val_roc = mean_scores

#         t = X_test["treatment"]
#         X_test_without_treatment = X_test.drop(columns=["treatment"])
#         X_test = torch.tensor(X_test_without_treatment.to_numpy(), dtype=torch.float32)
#         t_test = torch.tensor(t.to_numpy(), dtype=torch.float32)
#         y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)  # Drop 'treatment' from features
        

#         cevae = CEVAE(
#                 feature_dim=X_test.size(1), 
#                 num_layers=best_params["num_layers"],
#                 hidden_dim=best_params["hidden_dim"],
#                 num_samples=best_params["num_samples"])

#         losses = cevae.fit(
#                 X_test, 
#                 t_test, 
#                 y_test, 
#                 num_epochs=best_params["num_epochs"],
#                 learning_rate=best_params["learning_rate"],
#                 batch_size=200
#                 )


#         ites, y0, y1 = cevae.ite(X_test)
#         ate = torch.mean(torch.flatten(ites))
        
#         # flatten ys
#         y0 = y0.mean(dim=0).flatten()
#         y1 = y1.mean(dim=0).flatten()

#         # roc
#         # y0n = (torch.sigmoid(y0)>0.5).float()
#         # y1n = (torch.sigmoid(y1)>0.5).float()
#         y = torch.where(t_test > 0, y1, y0)

#         ROC = roc_auc_score(y_test.cpu(), y.cpu())

#         out_df = create_ate_df(ate.item(), ROC, combination, "CEVAE", best_params, best_val_roc)
#         results_df = pd.concat([results_df, out_df], ignore_index=True)

#     return pd.DataFrame(results_df)