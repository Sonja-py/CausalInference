import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from copy import deepcopy
import pickle
# from statistics import median, mean

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
# from causalml.inference.nn import CEVAE
# from pyro.contrib.cevae import CEVAE

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
    Output(rid="ri.foundry.main.dataset.720ebfa7-629e-4ae2-9d4d-e23ab6099284"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def Test_lr_tlearner(final_data):
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
    Output(rid="ri.foundry.main.dataset.3cbc3c8c-65b6-4f67-8c4e-40bc4da8bbe8"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def Test_rf_tlearner(final_data):
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
        return best_roc, best_ate, best_params

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(716968, 19080226), (739138, 703547)]
    # threshold = 0.4
    rocs_l = []
    rocs_r = []
    ates_r = []
    ates_l = []

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
    
        best_roc, best_ate, best_params = grid_search(X_train, y_train, t_train, X_valid, y_valid, t_valid, class_weight_dict, 'LR')
        print(f'ROC: {best_roc}, {best_params}')
        best_params_df = create_best_params_df(best_params, best_roc, best_ate, combination, 'LR')
        results_df = pd.concat([results_df, best_params_df], ignore_index=True)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    print('Total time taken:',datetime.now() - initial_time)

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

    def create_best_params_df(best_params, best_roc, best_ate, combination, model):
        best_params['val_roc'] = best_roc
        best_params['val_ate'] = best_ate
        best_params['drug_0'] = combination[0]
        best_params['drug_1'] = combination[1]
        best_params['model'] = model
        return pd.DataFrame(best_params, index=[0])

    def sample(datasetOfZippedFiles, filename):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        with fs.open(f'{filename}.pickle', mode='rb') as f:
            model = pickle.load(f)
        print(model)
        return model

    def temp(X_test, y_test, t_test, class_weight_dict, clf_learner):
        ate, ate_lower, ate_upper = clf_learner.estimate_ate(X=X_test,
                                                            treatment=t_test,
                                                            y=y_test,
                                                            return_ci=True,
                                                            bootstrap_ci=True,
                                                            n_bootstraps=50,
                                                            bootstrap_size=1000,
                                                            # pretrain=True,
                                                            )

        print(f'ATE: {ate}, lower: {ate_lower}, upper: {ate_upper}')
        return ate[0], ate_lower[0], ate_upper[0]

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame(columns=['ate', 'ate_lower', 'ate_upper', 'drug_0', 'drug_1', 'model'])
    # ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))[:2]
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
        
        clf_learner = sample(Test_lr_slearner, f'{combination[0]}_{combination[1]}')
        ate, ate_l, ate_u = temp(X_test, y_test, t_test, class_weight_dict, clf_learner)

        results_df.loc[-1] = [ate, ate_l, ate_u, combination[0], combination[1], 'S_LR']

    #     print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    #     to_pickle(best_model, f'{combination[0]}_{combination[1]}')
    # print('Total time taken:',datetime.now() - initial_time)

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
    initial_time = datetime.now()
    # ingredient_pairs = [(739138, 703547)]
    threshold = 0.4
    rocs_l = []
    rocs_r = []
    ates_r = []
    ates_l = []

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

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

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
    Output(rid="ri.vector.main.execute.f54ece32-1f97-4434-95c5-56c4d870300d"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def unnamed(final_data):
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.99ed3509-4e04-43f9-b117-5e5eb7c31098"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def unnamed_1(final_data):
    output = Transforms.get_output()
    output_fs = output.filesystem()
    val = []

    with output_fs.open('roc_r.txt', 'r') as f: 
        f.read('roc_r.txt')
        val.append(list(f))

    print(val)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.16efea9d-b80f-4579-8b8f-775e5fa2b60b")
)
def unnamed_2():
    # cevae = CEVAE(outcome_dist='binary')
    # # cevae
    # print(dir(cevae))

    # import argparse
    # import logging

    import torch

    import pyro
    import pyro.distributions as dist
    from pyro.contrib.cevae import CEVAE

    # logging.getLogger("pyro").setLevel(logging.DEBUG)
    # logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

    def generate_data():
        """
        This implements the generative process of [1], but using larger feature and
        latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
        """
        z = dist.Bernoulli(0.5).sample([100])
        x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([5]).t()
        t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
        y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

        # Compute true ite for evaluation (via Monte Carlo approximation).
        t0_t1 = torch.tensor([[0.0], [1.0]])
        y_t0, y_t1 = dist.Bernoulli(logits=3 * (z + 2 * (2 * t0_t1 - 2))).mean
        true_ite = y_t1 - y_t0
        return x, t, y, true_ite

    def main():
        # if args.cuda:
        #     torch.set_default_device("cuda")

        # Generate synthetic data.
        # pyro.set_rng_seed(args.seed)
        x_train, t_train, y_train, _ = generate_data()

        # Train.
        # pyro.set_rng_seed(args.seed)
        pyro.clear_param_store()
        # parser.add_argument("--num-data", default=1000, type=int)
    # parser.add_argument("--feature-dim", default=5, type=int)
    # parser.add_argument("--latent-dim", default=20, type=int)
    # parser.add_argument("--hidden-dim", default=200, type=int)
    # parser.add_argument("--num-layers", default=3, type=int)
        cevae = CEVAE(
            feature_dim=5,
            latent_dim=10,
            hidden_dim=10,
            num_layers=3,
            num_samples=10,
        )
        cevae.fit(
            x_train,
            t_train,
            y_train,
            num_epochs=5,
            batch_size=32,
            learning_rate=1e-3,
            # learning_rate_decay=args.learning_rate_decay,
            # weight_decay=args.weight_decay,
        )

        y_0 = cevae.model.y_mean(x_train, 0)
        y_1 = cevae.model.y_mean(x_train, 1)
        # print(y_train)
        y = (y_1 - y_0).mean(0)
        print(f'y is {y}')
        ite = cevae.ite(x_train)
        print(f'ite {ite}')
        # print(f'ite {y_1 - y_0}')

        # Evaluate.
        # x_test, t_test, y_test, true_ite = generate_data()
        # true_ate = true_ite.mean()
        # print("true ATE = {:0.3g}".format(true_ate.item()))
        # naive_ate = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        # print("naive ATE = {:0.3g}".format(naive_ate))
        # if args.jit:
        #     cevae = cevae.to_script_module()
        # est_ite = cevae.ite(x_test)
        # est_ate = est_ite.mean()
        # print("estimated ATE = {:0.3g}".format(est_ate.item()))

    # assert pyro.__version__.startswith("1.9.0")
    # parser = argparse.ArgumentParser(
    #     description="Causal Effect Variational Autoencoder"
    # )
    # parser.add_argument("--num-data", default=1000, type=int)
    # parser.add_argument("--feature-dim", default=5, type=int)
    # parser.add_argument("--latent-dim", default=20, type=int)
    # parser.add_argument("--hidden-dim", default=200, type=int)
    # parser.add_argument("--num-layers", default=3, type=int)
    # parser.add_argument("-n", "--num-epochs", default=50, type=int)
    # parser.add_argument("-b", "--batch-size", default=100, type=int)
    # parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    # parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    # parser.add_argument("--seed", default=1234567890, type=int)
    # parser.add_argument("--jit", action="store_true")
    # parser.add_argument("--cuda", action="store_true")
    # args = parser.parse_args()

    main()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cece95e1-548c-4d0f-95c5-9240d885dec0"),
    Test_lr_slearner=Input(rid="ri.foundry.main.dataset.67236741-6d93-418d-83c3-91a2b3ea8405")
)
def unnamed_3(Test_lr_slearner):

    import tempfile
    import zipfile
    import shutil
    import io
    from pyspark.sql import Row

    # datasetOfZippedFiles is a dataset with a single zipped file that contains 3 CSVs with the same schema: ["id", name"].
    def sample(datasetOfZippedFiles):
        df = datasetOfZippedFiles
        fs = df.filesystem() # This is the FileSystem object.
        
        with fs.open('40234834_710062.pickle', mode='rb') as f:
            model = pickle.load(f)
        print(model)
        print(str(model))

        ate, ate_lower, ate_upper = model.estimate_ate(X=X_test,
                                                    treatment=t_test,
                                                    y=y_test,
                                                    return_ci=True,
                                                    bootstrap_ci=True,
                                                    n_bootstraps=50,
                                                    bootstrap_size=1000,
                                                    pretrain=True,)
        # return rdd

    sample(Test_lr_slearner)

#     import os
#     output = Transforms.get_output()
#     output_fs = output.filesystem()
# # /UNITE/[RP-1225E6] Effects of drugs on COVID-19 trajectory/Anugrah Analysis/workbook-output/Model_Training_Meta_Learners/Test lr slearner
#     with output_fs.open('Test_lr_slearner/40234834_710062.pickle', 'rb') as f:
#         print(os.getcwd())
#         data = pickle.load(f)
#         print('Load done')
    
#     print(data.best_params_)
#     print(type(data))
#     print(data.keys())

#     return data

