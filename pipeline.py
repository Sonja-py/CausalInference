import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from copy import deepcopy
import pickle
from pyspark.sql import DataFrame
from functools import reduce
from statistics import median, mean

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from causalml.inference.nn import CEVAE
from pyro.contrib.cevae import CEVAE

from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold

# Save error metrics
def write_text_file(data, metric):
    output = Transforms.get_output()
    output_fs = output.filesystem()
    metric, learner = metric.split('_')
    
    if metric == 'rocs' and learner == 'r':
        filename = 'roc_r.txt'
    elif metric == 'rocs' and learner == 'l':
        filename = 'roc_l.txt'
    elif metric == 'ates' and learner == 'r':
        filename = 'ate_r.txt'
    else:
        filename = 'ate_l.txt'
    with output_fs.open(filename, 'w') as f: 
        f.write(str(data))
   

# Save GS model
def to_pickle(data, filename):
    output = Transforms.get_output()
    output_fs = output.filesystem()
    
    with output_fs.open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(data, f)

        

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aa8fcdda-8570-4c04-b0d5-3b1afa7d04e6"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def cevae(final_data):

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    ingredient_list = main_df.ingredient_concept_id.unique()[:2]
    ingredient_pairs = list(combinations(ingredient_list, 2))
    # rocs = []
    ates = []

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running CEVAE for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df[main_df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        # np.random.seed(3)

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.2, random_state = 42, stratify = y_test)

        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Parameter grid (you should customize this)
        param_grid = [
            {'dim_hidden': 20, 'dim_latent': 5, 'num_layers': 2},
            {'dim_hidden': 20, 'dim_latent': 5, 'num_layers': 3},
            # Add more parameter combinations as needed
        ]

        # Best model score and parameters
        best_score = 0
        best_params = None

        # Loop over each fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            t_train, t_test = t[train_index], t[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Inner loop: iterate over all possible combinations of parameters
            for params in param_grid:
                # Initialize and fit the CEVAE model with current set of parameters
                cevae = CEVAE(outcome_dist="binary", **params)
                cevae.fit(X=X_train, treatment=t_train, y=y_train)
                
                # Predict the causal effect on the validation data
                y_pred = cevae.predict(X_test)
                
                # Calculate ROC AUC score
                score = roc_auc_score(y_test, y_pred)
                
                # Update best score and parameters if current model is better
                if score > best_score:
                    best_score = score
                    best_params = params

        # Print out the best parameter set and its performance
        print(f'Best parameters: {best_params}')
        print(f'Best ROC AUC score: {best_score}')
        t_train = t[X_train.index]
        t_test = t[X_test.index]
        t_valid = t[X_valid.index]

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        print('Class weights dict', class_weight_dict)

        cevae_model = CEVAE(num_epochs = 10, batch_size = 128, learning_rate = 1e-2, num_samples = 100)
        cevae_model.fit(X=X_train, treatment=t_train, y=y_train)
        
        ite = cevae_model.predict(X_valid.to_numpy())
        ate = ite.mean()
        print('ATE:',ate)
        ates.append(ate)
        # save_model(cevae_model, str(combination[0]) + '_' + str(combination[1]))
        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')
        

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.275b5e62-ddd3-435c-a92f-4fe2a9da8c33"),
    combined_hyperparams=Input(rid="ri.foundry.main.dataset.0d645ea3-8041-482e-a548-ea708421e06b"),
    ds16=Input(rid="ri.foundry.main.dataset.02565d66-6582-40ef-b528-c2f3d2f4925f")
)
def combined_hyperparameters(combined_hyperparams, ds16):
    dfs = [combined_hyperparams, ds16]
    df = reduce(DataFrame.unionAll, dfs)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.39b1c667-50bf-4352-adb1-b33a3ba8ac47"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def lr_slearner(final_data):
    import warnings
    warnings.filterwarnings('ignore')

    def metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc = roc_auc_score(y_valid, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, best_yhat, best_that, model):
        best_params['roc'] = best_roc
        best_params['ate'] = best_ate
        best_params['y_hat'] = best_yhat
        best_params['t_hat'] = best_that
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        best_yhat = 0.0
        best_that = 0.0
        l1_ratio = None
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
        # estim_1 = ['elasticnet'] # penalty
        estim_2 = [0, 0.25, 0.5, 0.75, 1] # l1_ratio
        # estim_3 = [100, 500, 1000] # max_iter
        estim_4 = [0.01, 0.1, 1, 10, 100] # C - regularization strength
        # for crit_1 in estim_1:
        for crit_2 in estim_2:
            # for crit_3 in estim_3:
            for crit_4 in estim_4:
                clf = LogisticRegression(penalty='elasticnet', l1_ratio=crit_2, max_iter=100, C=crit_4, solver='saga', class_weight=class_weight_dict)
                clf_learner = BaseSClassifier(learner = clf)
                clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
                roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts)
                
                if roc > best_roc:
                    best_ate = ate
                    best_roc = roc
                    best_yhat = yhat_cs
                    best_that = yhat_ts
                    # best_params = {'parameters': [('n_estimators', estimator), ('criterion', criterion), ('max_depth', depth)]}
                    best_params = {'n_estimators': np.nan, 
                                    'criterion': np.nan,
                                    'max_depth': np.nan,
                                    'l1_ratio':crit_2,
                                    'C':crit_4,
                                    # 'max_iter':crit_3,
                                    # 'solver':crit_4,
                                    }
                # print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')
                # print(f'Done - penalty: {crit_1}, C: {crit_2}, max_iter: {crit_3}, solver: {crit_4}')
        return best_roc, best_ate, best_params, best_yhat, best_that

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    # initial_time = datetime.now()
    # ingredient_pairs = [(716968, 19080226), (739138, 703547)]
    # threshold = 0.4
    rocs_l = []
    rocs_r = []
    ates_r = []
    ates_l = []

    for idx, combination in enumerate(ingredient_pairs):
        # start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        t_train = t[X_train.index]
        t_train = t_train.values
        t_test = t[X_test.index]
        t_test = t_test.values
        t_valid = t[X_valid.index]
        t_valid = t_valid.values

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
    
        best_roc, best_ate, best_params, best_yhat, best_that = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, {best_params}')
        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, best_yhat, best_that, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        # print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    # print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2c30c167-7995-4e2a-849d-9fb7c5a109d2"),
    Test_lr_slearner=Input(rid="ri.foundry.main.dataset.67236741-6d93-418d-83c3-91a2b3ea8405"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def lr_slearner_bootstrap(final_data, Test_lr_slearner):
    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(ate, ate_lower, ate_upper, variance, combination, model):
        best_params = {}
        best_params['ate'] = ate
        best_params['ate_lower'] = ate_lower
        best_params['ate_upper'] = ate_upper
        best_params['variance'] = variance
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        print(df.filesystem().ls())
        try:
            with fs.open(f"{filename[0]}.pkl", mode="rb") as f:
                model = pickle.load(f)
                return model
        except Exception as e:
            with fs.open(f"{filename[1]}.pkl", mode="rb") as f:
                model = pickle.load(f)
                return model

    def temp(X_test, y_test, t_test, bs_size, class_weight_dict, clf_learner):
        ate, ate_lower, ate_upper = clf_learner.estimate_ate(X=X_test,
                                                            treatment=t_test,
                                                            y=y_test,
                                                            return_ci=True,
                                                            bootstrap_ci=True,
                                                            n_bootstraps=100,
                                                            bootstrap_size=bs_size,
                                                            # pretrain=True,
                                                            )

        print(f'ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}')
        return ate[0], ate_lower[0], ate_upper[0]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'variance', 'drug_0', 'drug_1', 'model'])
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(40234834, 710062)]

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        
        clf_learner = sample(Test_lr_slearner, [
                f"{combination[0]}_{combination[1]}",
                f"{combination[1]}_{combination[0]}",
            ])
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, X_train_val.shape[0], class_weight_dict, clf_learner)

        # results_df.loc[-1] = [ate, ate_l, ate_u, ate_u - ate_l, combination[0], combination[1], 'S_LR']
        params_df = create_best_params_df(ate, ate_l, ate_u, ate_u - ate_l, combination, 'S_LR')
        results_df = pd.concat([results_df, params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.492668a1-6094-4392-9901-c5b86103fb69"),
    Test_lr_tlearner=Input(rid="ri.foundry.main.dataset.720ebfa7-629e-4ae2-9d4d-e23ab6099284"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def lr_tlearner_bootstrap(final_data, Test_lr_tlearner):
    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(ate, ate_lower, ate_upper, variance, combination, model):
        best_params = {}
        best_params['ate'] = ate
        best_params['ate_lower'] = ate_lower
        best_params['ate_upper'] = ate_upper
        best_params['variance'] = variance
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        try:
            with fs.open(f"{filename[0]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model
        except Exception as e:
            with fs.open(f"{filename[1]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model

    def temp(X_test, y_test, t_test, bs_size, class_weight_dict, clf_learner):
        ate, ate_lower, ate_upper = clf_learner.estimate_ate(X=X_test,
                                                            treatment=t_test,
                                                            y=y_test,
                                                            # return_ci=True,
                                                            bootstrap_ci=True,
                                                            n_bootstraps=100,
                                                            bootstrap_size=bs_size,
                                                            # pretrain=True,
                                                            )

        print(f'ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}')
        return ate[0], ate_lower[0], ate_upper[0]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'variance', 'drug_0', 'drug_1', 'model'])
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(40234834, 710062)]

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        
        clf_learner = sample(Test_lr_tlearner, [
                f"{combination[0]}_{combination[1]}",
                f"{combination[1]}_{combination[0]}",
            ])
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, X_train_val.shape[0], class_weight_dict, clf_learner)

        # results_df.loc[-1] = [ate, ate_l, ate_u, ate_u - ate_l, combination[0], combination[1], 'S_LR']
        params_df = create_best_params_df(ate, ate_l, ate_u, ate_u - ate_l, combination, 'T_LR')
        results_df = pd.concat([results_df, params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.f2cebbba-3c15-4e6c-b89b-d8374a3b91f3"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def meta_learners_bootstrapped(final_data):

    df = final_data.toPandas()
    df =  df[df['age_at_covid'].notna()]
    df = df.reset_index(drop=True)

    X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
    y = df['severity_final']
    t = df['treatment']

    np.random.seed(3)

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights dict', class_weight_dict)

    model_t1 = RandomForestClassifier(n_estimators = 500, max_depth = 20, class_weight = class_weight_dict)
    learner_t1 = BaseTClassifier(learner = model_t1)
    ate_t1 = learner_t1.estimate_ate(X=X, treatment=t, y=y, n_bootstraps=10)
    # cate_t1 = learner_t1.fit_predict(X=X, treatment=t, y=y, n_bootstraps=10)
    print(f'CATE T-Learner: RandomForest - Mean {ate_t1[0]}, LB {ate_t1[1]}, UB {ate_t1[2]}')

    model_t2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    learner_t2 = BaseTClassifier(learner = model_t2)
    cate_t2 = learner_t2.estimate_ate(X=X, treatment=t, y=y, n_bootstraps=10)
    print(f'CATE T-Learner: LogisticRegression - Mean {ate_t1[0]}, LB {ate_t1[1]}, UB {ate_t1[2]}')

    model_s1 = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight = class_weight_dict)
    learner_s1 = BaseSClassifier(learner = model_s1)
    cate_s1 = learner_s1.estimate_ate(X=X, treatment=t, y=y, n_bootstraps=10)
    print('CATE S-Learner: RandomForest', cate_s1)

    model_s2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    learner_s2 = BaseSClassifier(learner = model_s2)
    cate_s2 = learner_s2.estimate_ate(X=X, treatment=t, y=y, n_bootstraps=10)
    print('CATE S-Learner: LogisticRegression', cate_s2)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c55c9d7f-ff97-450b-8d2c-fabecfaaa8ab"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from itertools import combinations
from sklearn.base import BaseEstimator
from causalml.inference.meta import BaseSClassifier

def rf_slearner(final_data):

    def metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc = roc_auc_score(y_valid, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['roc'] = best_roc
        best_params['ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        estim_1 = [100, 500, 1000] # n_estimators
        estim_3 = [None, 25, 50, 75, 100] # max_depth
        for crit_1 in estim_1:
            for crit_3 in estim_3:
                clf = RandomForestClassifier(n_estimators = crit_1, criterion = 'log_loss', max_depth = crit_3, class_weight=class_weight_dict)
                clf_learner = BaseSClassifier(learner = clf)
                clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
                roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts)
                
                if roc > best_roc:
                    best_ate = ate
                    best_roc = roc
                    best_params = {'n_estimators': crit_1, 'max_depth': crit_3, 'penalty':np.nan, 'C':np.nan, 'max_iter':np.nan, 'solver':np.nan}
        return best_roc, best_ate, best_params

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()[:2]
    ingredient_pairs = list(combinations(ingredient_list, 2))
    # initial_time = datetime.now()
    # ingredient_pairs = [(739138, 703547)]
    threshold = 0.4
    rocs_l = []
    rocs_r = []
    ates_r = []
    ates_l = []

    for idx, combination in enumerate(ingredient_pairs):
        # start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        t_train = t[X_train.index]
        t_train = t_train.values
        t_test = t[X_test.index]
        t_test = t_test.values
        t_valid = t[X_valid.index]
        t_valid = t_valid.values

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'RF')
        print(f'ROC: {best_roc}, {best_params}')
        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'RF')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        # print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    # print('Total time taken:',datetime.now() - initial_time)

    return results_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5d31a36e-3779-4484-8d2a-7436ca0d827f"),
    Test_rf_slearner=Input(rid="ri.foundry.main.dataset.d8a3ce5b-5472-4704-9d3a-205920048c80"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def rf_slearner_bootstrap(final_data, Test_rf_slearner):
    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(ate, ate_lower, ate_upper, variance, combination, model):
        best_params = {}
        best_params['ate'] = ate
        best_params['ate_lower'] = ate_lower
        best_params['ate_upper'] = ate_upper
        best_params['variance'] = variance
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        try:
            with fs.open(f"{filename[0]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model
        except Exception as e:
            with fs.open(f"{filename[1]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model

    def temp(X_test, y_test, t_test, bs_size, class_weight_dict, clf_learner):
        ate, ate_lower, ate_upper = clf_learner.estimate_ate(X=X_test,
                                                            treatment=t_test,
                                                            y=y_test,
                                                            return_ci=True,
                                                            bootstrap_ci=True,
                                                            n_bootstraps=100,
                                                            bootstrap_size=bs_size,
                                                            # pretrain=True,
                                                            )

        print(f'ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}')
        return ate[0], ate_lower[0], ate_upper[0]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'variance', 'drug_0', 'drug_1', 'model'])
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(40234834, 710062)]

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        
        clf_learner = sample(Test_rf_slearner, [
                f"{combination[0]}_{combination[1]}",
                f"{combination[1]}_{combination[0]}",
            ])
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, X_train_val.shape[0], class_weight_dict, clf_learner)

        # results_df.loc[-1] = [ate, ate_l, ate_u, ate_u - ate_l, combination[0], combination[1], 'S_LR']
        params_df = create_best_params_df(ate, ate_l, ate_u, ate_u - ate_l, combination, 'S_RF')
        results_df = pd.concat([results_df, params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f659a7c1-d747-471d-94d5-80a85fb8e1c3"),
    Test_rf_tlearner=Input(rid="ri.foundry.main.dataset.3cbc3c8c-65b6-4f67-8c4e-40bc4da8bbe8"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from itertools import combinations
from sklearn.base import BaseEstimator
from causalml.inference.meta import BaseSClassifier
def rf_tlearner_bootstrap(final_data, Test_rf_tlearner):
    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(ate, ate_lower, ate_upper, variance, combination, model):
        best_params = {}
        best_params['ate'] = ate
        best_params['ate_lower'] = ate_lower
        best_params['ate_upper'] = ate_upper
        best_params['variance'] = variance
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        try:
            with fs.open(f"{filename[0]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model
        except Exception as e:
            with fs.open(f"{filename[1]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model

    def temp(X_test, y_test, t_test, bs_size, class_weight_dict, clf_learner):
        ate, ate_lower, ate_upper = clf_learner.estimate_ate(X=X_test,
                                                            treatment=t_test,
                                                            y=y_test,
                                                            # return_ci=True,
                                                            bootstrap_ci=True,
                                                            n_bootstraps=100,
                                                            bootstrap_size=bs_size,
                                                            # pretrain=True,
                                                            )

        print(f'ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}')
        return ate[0], ate_lower[0], ate_upper[0]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'variance', 'drug_0', 'drug_1', 'model'])
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(40234834, 710062)]

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        
        clf_learner = sample(Test_rf_tlearner, [
                f"{combination[0]}_{combination[1]}",
                f"{combination[1]}_{combination[0]}",
            ])
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, X_train_val.shape[0], class_weight_dict, clf_learner)

        # results_df.loc[-1] = [ate, ate_l, ate_u, ate_u - ate_l, combination[0], combination[1], 'S_LR']
        params_df = create_best_params_df(ate, ate_l, ate_u, ate_u - ate_l, combination, 'T_RF')
        results_df = pd.concat([results_df, params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.67236741-6d93-418d-83c3-91a2b3ea8405"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def test_lr_slearner(final_data):
    import warnings
    warnings.filterwarnings('ignore')

    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['val_roc'] = best_roc
        best_params['val_ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        best_model = None
        for idx, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
            # Generate training and validation sets for the fold
            X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            t_train, t_val = t_train_val.iloc[train_index], t_train_val.iloc[val_index]

            estim_2 = [0, 0.25, 0.5, 0.75, 1] # l1_ratio
            estim_4 = [0.01, 0.1, 1, 10, 100] # C - regularization strength
            for crit_2 in estim_2:
                for crit_4 in estim_4:
                    clf = LogisticRegression(penalty='elasticnet', l1_ratio=crit_2, max_iter=100, C=crit_4, solver='saga', class_weight=class_weight_dict)
                    clf_learner = BaseSClassifier(learner = clf)
                    clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                    ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_val, treatment=t_val, y=y_val, return_components=True, verbose=True)
                    roc, ate = metrics(y_val, t_val, ite, yhat_cs, yhat_ts)
                    
                    if roc > best_roc:
                        best_model = clf_learner
                        best_ate = ate
                        best_roc = roc
                        best_params = {'n_estimators': np.nan, 
                                        'criterion': np.nan,
                                        'max_depth': np.nan,
                                        'l1_ratio':crit_2,
                                        'C':crit_4,
                                        }
                    # print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')

        return best_roc, best_ate, best_params, best_model

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(716968, 19080226), (739138, 703547)]
    # threshold = 0.4

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params, best_model = grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, 'LR')

        # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        # y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        # t_train = t[X_train.index]
        # t_train = t_train.values
        # t_test = t[X_test.index]
        # t_test = t_test.values
        # t_valid = t[X_valid.index]
        # t_valid = t_valid.values

        # best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, ATE: {best_ate}, {best_params}')

        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

        to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.720ebfa7-629e-4ae2-9d4d-e23ab6099284"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def test_lr_tlearner(final_data):
    import warnings
    warnings.filterwarnings('ignore')

    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['val_roc'] = best_roc
        best_params['val_ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        best_model = None
        for idx, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
            # Generate training and validation sets for the fold
            X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            t_train, t_val = t_train_val.iloc[train_index], t_train_val.iloc[val_index]

            estim_2 = [0, 0.25, 0.5, 0.75, 1] # l1_ratio
            estim_4 = [0.01, 0.1, 1, 10, 100] # C - regularization strength
            for crit_2 in estim_2:
                for crit_4 in estim_4:
                    clf = LogisticRegression(penalty='elasticnet', l1_ratio=crit_2, max_iter=100, C=crit_4, solver='saga', class_weight=class_weight_dict)
                    clf_learner = BaseTClassifier(learner = clf)
                    clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                    ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_val, treatment=t_val, y=y_val, return_components=True, verbose=True)
                    roc, ate = metrics(y_val, t_val, ite, yhat_cs, yhat_ts)
                    
                    if roc > best_roc:
                        best_model = clf_learner
                        best_ate = ate
                        best_roc = roc
                        best_params = {'n_estimators': np.nan, 
                                        'criterion': np.nan,
                                        'max_depth': np.nan,
                                        'l1_ratio':crit_2,
                                        'C':crit_4,
                                        }
                    # print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')

        return best_roc, best_ate, best_params, best_model

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(716968, 19080226), (739138, 703547)]
    # threshold = 0.4

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params, best_model = grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, 'LR')

        # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        # y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        # t_train = t[X_train.index]
        # t_train = t_train.values
        # t_test = t[X_test.index]
        # t_test = t_test.values
        # t_valid = t[X_valid.index]
        # t_valid = t_valid.values

    
        # best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, {best_params}')

        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

        to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d8a3ce5b-5472-4704-9d3a-205920048c80"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def test_rf_slearner(final_data):
    import warnings
    warnings.filterwarnings('ignore')

    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['val_roc'] = best_roc
        best_params['val_ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        best_model = None
        for idx, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
            # Generate training and validation sets for the fold
            X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            t_train, t_val = t_train_val.iloc[train_index], t_train_val.iloc[val_index]

            estim_1 = [2, 5] # min_samples_split
            # estim_2 = [50, 100] # n_estimators
            estim_3 = [1, 5, 10] # min_samples_leaf
            estim_4 = [None, 25, 50] # max_depth
            for crit_1 in estim_1:
                for crit_3 in estim_3:
                    for crit_4 in estim_4:
                        clf = RandomForestClassifier(min_samples_split=crit_1, min_samples_leaf=crit_3, n_estimators=100, criterion='log_loss', max_depth=crit_4, class_weight=class_weight_dict)
                        clf_learner = BaseSClassifier(learner = clf)
                        clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                        ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_val, treatment=t_val, y=y_val, return_components=True, verbose=True)
                        roc, ate = metrics(y_val, t_val, ite, yhat_cs, yhat_ts)
                        
                        if roc > best_roc:
                            best_model = clf_learner
                            best_ate = ate
                            best_roc = roc
                            best_params = {'min_samples_split': crit_1, 
                                            'min_samples_leaf': crit_3,
                                            'max_depth': crit_4,
                                            'l1_ratio':np.nan,
                                            'C':np.nan,
                                            }
                        # print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')

        return best_roc, best_ate, best_params, best_model

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params, best_model = grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, 'LR')

        # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        # y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        # t_train = t[X_train.index]
        # t_train = t_train.values
        # t_test = t[X_test.index]
        # t_test = t_test.values
        # t_valid = t[X_valid.index]
        # t_valid = t_valid.values

    
        # best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, {best_params}')

        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

        to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3cbc3c8c-65b6-4f67-8c4e-40bc4da8bbe8"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def test_rf_tlearner(final_data):
    import warnings
    warnings.filterwarnings('ignore')

    def metrics(y, t, ite, yhat_cs, yhat_ts):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t) * yhat_cs + t * yhat_ts
        roc = roc_auc_score(y, preds)
        ate = ite.mean()
        return roc, ate

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['val_roc'] = best_roc
        best_params['val_ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, model):
        best_roc = 0.0
        best_ate = 0.0
        best_model = None
        for idx, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
            # Generate training and validation sets for the fold
            X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            t_train, t_val = t_train_val.iloc[train_index], t_train_val.iloc[val_index]

            estim_1 = [2, 5] # min_samples_split
            # estim_2 = [50, 100] # n_estimators
            estim_3 = [1, 5, 10] # min_samples_leaf
            estim_4 = [None, 25, 50] # max_depth
            for crit_1 in estim_1:
                for crit_3 in estim_3:
                    for crit_4 in estim_4:
                        clf = RandomForestClassifier(min_samples_split=crit_1, min_samples_leaf=crit_3, n_estimators=100, criterion='log_loss', max_depth=crit_4, class_weight=class_weight_dict)
                        clf_learner = BaseTClassifier(learner = clf)
                        clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                        ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_val, treatment=t_val, y=y_val, return_components=True, verbose=True)
                        roc, ate = metrics(y_val, t_val, ite, yhat_cs, yhat_ts)
                        
                        if roc > best_roc:
                            best_model = clf_learner
                            best_ate = ate
                            best_roc = roc
                            best_params = {'min_samples_split': crit_1, 
                                            'min_samples_leaf': crit_3,
                                            'max_depth': crit_4,
                                            'l1_ratio':np.nan,
                                            'C':np.nan,
                                            }
                        # print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')

        return best_roc, best_ate, best_params, best_model

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df.copy()
        df = df[df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(X, y, t, test_size=0.2, random_state=42, stratify=y)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))

        best_roc, best_ate, best_params, best_model = grid_search(X_train_val, y_train_val, t_train_val, skf, class_weight_dict, 'LR')

        # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, stratify = y_test)
        # y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        
        # t_train = t[X_train.index]
        # t_train = t_train.values
        # t_test = t[X_test.index]
        # t_test = t_test.values
        # t_valid = t[X_valid.index]
        # t_valid = t_valid.values

    
        # best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, {best_params}')

        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

        to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    print('Total time taken:',datetime.now() - initial_time)

    return results_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.3cc31dfe-610e-492d-924d-c5f6421324d1"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def testing(final_data):

    # def fit(X, treatment, y, learner, p=None):
    #     """Fit the inference model

    #     Args:
    #         X (np.matrix or np.array or pd.Dataframe): a feature matrix
    #         treatment (np.array or pd.Series): a treatment vector
    #         y (np.array or pd.Series): an outcome vector
    #     """
    #     model_c = deepcopy(learner)
    #     model_t = deepcopy(learner)
    #     control_name = 0
    #     X, treatment, y = convert_pd_to_np(X, treatment, y)
    #     check_treatment_vector(treatment, control_name)
    #     t_groups = np.unique(treatment[treatment != control_name])
    #     t_groups.sort()
    #     _classes = {group: i for i, group in enumerate(t_groups)}
    #     models_c = {group: deepcopy(model_c) for group in t_groups}
    #     models_t = {group: deepcopy(model_t) for group in t_groups}

    #     for group in t_groups:
    #         mask = (treatment == group) | (treatment == control_name)
    #         treatment_filt = treatment[mask]
    #         X_filt = X[mask]
    #         y_filt = y[mask]
    #         w = (treatment_filt == group).astype(int)

    #         models_c[group].fit(X_filt[w == 0], y_filt[w == 0])
    #         models_t[group].fit(X_filt[w == 1], y_filt[w == 1])

    # def predict(
    #     X, treatment=None, y=None, p=None, return_components=False, verbose=True
    # ):
    #     """Predict treatment effects.

    #     Args:
    #         X (np.matrix or np.array or pd.Dataframe): a feature matrix
    #         treatment (np.array or pd.Series, optional): a treatment vector
    #         y (np.array or pd.Series, optional): an outcome vector
    #         verbose (bool, optional): whether to output progress logs
    #     Returns:
    #         (numpy.ndarray): Predictions of treatment effects.
    #     """
    #     yhat_cs = {}
    #     yhat_ts = {}

    #     for group in t_groups:
    #         model_c = models_c[group]
    #         model_t = models_t[group]
    #         yhat_cs[group] = model_c.predict_proba(X)[:, 1]
    #         yhat_ts[group] = model_t.predict_proba(X)[:, 1]

    #         if (y is not None) and (treatment is not None) and verbose:
    #             mask = (treatment == group) | (treatment == control_name)
    #             treatment_filt = treatment[mask]
    #             y_filt = y[mask]
    #             w = (treatment_filt == group).astype(int)

    #             yhat = np.zeros_like(y_filt, dtype=float)
    #             yhat[w == 0] = yhat_cs[group][mask][w == 0]
    #             yhat[w == 1] = yhat_ts[group][mask][w == 1]

    #             logger.info("Error metrics for group {}".format(group))
    #             classification_metrics(y_filt, yhat, w)

    #     te = np.zeros((X.shape[0], t_groups.shape[0]))
    #     for i, group in enumerate(t_groups):
    #         te[:, i] = yhat_ts[group] - yhat_cs[group]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    ingredient_list = main_df.ingredient_concept_id.unique()[:2]
    ingredient_pairs = list(combinations(ingredient_list, 2))
    # rocs = []
    ates = []

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df[main_df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.7, random_state = 42, stratify = y_test)

        y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values
        t_train = t[X_train.index]
        t_train = t_train.values
        t_test = t[X_test.index]
        t_test = t_test.values
        t_valid = t[X_valid.index]
        t_valid = t_valid.values
        print('shape of valid data:',t_valid.shape)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        print('Class weights dict', class_weight_dict)

        modelt1 = RandomForestClassifier(n_estimators = 100, max_depth = 15, class_weight = class_weight_dict)
        learner_t1 = BaseTClassifier(learner = modelt1)
        learner_t1.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_t1.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)

        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc_score = roc_auc_score(y_valid, preds)
        print('ATE:',ite.mean())
        print('ROC score:', roc_score)
        

@transform_pandas(
    Output(rid="ri.vector.main.execute.358a4ee9-2f0f-4be6-bec9-b5d34727750f"),
    Test_lr_slearner=Input(rid="ri.foundry.main.dataset.67236741-6d93-418d-83c3-91a2b3ea8405"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def unnamed(Test_lr_slearner, final_data):
    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        try:
            with fs.open(f"{filename[0]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model
        except Exception as e:
            with fs.open(f"{filename[1]}.pickle", mode="rb") as f:
                model = pickle.load(f)
                return model

    
    main_df = final_data.toPandas()
    # results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'variance', 'drug_0', 'drug_1', 'model'])
    # ingredient_list = main_df.ingredient_concept_id.unique()
    # ingredient_pairs = list(combinations(ingredient_list, 2))
    ingredient_pairs = [(40234834, 710062)]
    for idx, combination in enumerate(ingredient_pairs):
        model = sample(Test_lr_slearner, [
                    f"{combination[0]}_{combination[1]}",
                    f"{combination[1]}_{combination[0]}",
                ])

        break
    print(model)
    return model

@transform_pandas(
    Output(rid="ri.vector.main.execute.7fbdc1e1-5256-4228-a479-d060987c67c6")
)
from pyspark.sql.types import *
def unnamed_1():
    schema = StructType([])
    return spark.createDataFrame([[]], schema=schema)

@transform_pandas(
    Output(rid="ri.vector.main.execute.d8bccc7a-6fb3-4024-a64e-dcd15a852ce9")
)
from pyspark.sql.types import *
def unnamed_2():
    schema = StructType([])
    return spark.createDataFrame([[]], schema=schema)

@transform_pandas(
    Output(rid="ri.vector.main.execute.89021ed6-53c2-4027-8855-4b7e05f30b16"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def unnamed_4(final_data):
    source_df = final_data.toPandas()
    ingredient_list = source_df.ingredient_concept_id.unique()
    ingredient_pairs = sorted(list(combinations(ingredient_list, 2)), reverse=False)
    
    list_combs = [(703547, 725131),
    (703547, 738156),
    (703547, 743670),
    (703547, 750982),
    (703547, 710062),
    (703547, 715259),
    (703547, 715939),
    (703547, 717607),
    (703547, 721724),
    (703547, 722031),
    (710062, 722031),
    (715259, 710062),
    (715259, 721724),
    (715259, 722031),
    (703547, 40234834),
    (703547, 44507700),
    (703547, 751412),
    (703547, 755695),
    (703547, 778268),
    (703547, 797617),
    (715259, 797617),
    (715259, 40234834),
    (715259, 44507700),
    (715939, 710062),
    (715259, 725131),
    (715259, 738156),
    (715259, 755695),
    (715259, 778268),
    (715259, 743670),
    (715259, 751412),
    (715939, 750982),
    (715939, 751412),
    (715939, 755695),
    (715939, 778268),
    (715939, 715259),
    (715939, 721724),
    (715939, 738156),
    (715939, 743670),
    (715939, 722031),
    (715939, 725131),
    (717607, 721724),
    (717607, 722031),
    (717607, 725131),
    (717607, 738156),
    (715939, 44507700),
    (717607, 710062),
    (717607, 715259),
    (717607, 715939),
    (715939, 797617),
    (715939, 40234834),
    (717607, 40234834),
    (717607, 44507700),
    (721724, 710062),
    (721724, 722031),
    (717607, 743670),
    (717607, 750982),
    (717607, 751412),
    (717607, 755695),
    (717607, 778268),
    (717607, 797617),
    (738156, 721724),
    (738156, 722031),
    (738156, 725131),
    (738156, 743670),
    (725131, 721724),
    (725131, 722031),
    (721724, 778268),
    (725131, 710062),
    (725131, 778268),
    (738156, 710062),
    (739138, 715259),
    (739138, 715939),
    (739138, 717607),
    (739138, 721724),
    (738156, 751412),
    (738156, 778268),
    (738156, 797617),
    (738156, 40234834),
    (739138, 703547),
    (739138, 710062),
    (739138, 755695),
    (739138, 778268),
    (739138, 797617),
    (739138, 40234834),
    (739138, 722031),
    (739138, 725131),
    (739138, 738156),
    (739138, 743670),
    (739138, 750982),
    (739138, 751412),
    (743670, 778268),
    (743670, 797617),
    (743670, 40234834),
    (750982, 710062),
    (743670, 725131),
    (743670, 751412),
    (743670, 721724),
    (743670, 722031),
    (739138, 44507700),
    (743670, 710062),
    (750982, 751412),
    (750982, 755695),
    (750982, 778268),
    (750982, 797617),
    (750982, 722031),
    (750982, 725131),
    (750982, 715259),
    (750982, 721724),
    (750982, 738156),
    (750982, 743670),
    (751412, 778268),
    (751412, 797617),
    (755695, 710062),
    (755695, 721724),
    (751412, 710062),
    (751412, 721724),
    (751412, 722031),
    (751412, 725131),
    (750982, 40234834),
    (750982, 44507700),
    (755695, 797617),
    (755695, 40234834),
    (755695, 44507700),
    (778268, 710062),
    (755695, 738156),
    (755695, 743670),
    (755695, 751412),
    (755695, 778268),
    (755695, 722031),
    (755695, 725131),
    (40234834, 710062),
    (40234834, 721724),
    (40234834, 722031),
    (40234834, 725131),
    (797617, 721724),
    (797617, 722031),
    (797617, 725131),
    (797617, 778268),
    (778268, 722031),
    (797617, 710062),
    (44507700, 725131),
    (44507700, 738156),
    (44507700, 743670),
    (44507700, 751412),
    (40234834, 797617),
    (44507700, 710062),
    (40234834, 751412),
    (40234834, 778268),
    (44507700, 721724),
    (44507700, 722031)]

    # lst = []
    # for val in list_combs:
    #     print(val)
    #     if val not in ingredient_pairs:
    #        lst.append(val) 

    set_ing = set(ingredient_pairs)
    set_list = set(list_combs)
    print(len(set_ing), set_ing)
    print(len(set_list), set_list)
    print(set_ing - set_list)
    print(set_list - set_ing)
    
    # print(lst)

