import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from copy import deepcopy
# from statistics import median, mean

from causalml.inference.meta import BaseSClassifier, BaseTClassifier
from causalml.inference.nn import CEVAE

from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    

# Save model
def save_model(model, output_filename):
    output = Transforms.get_output()
    output_fs = output.filesystem()
    
    with output_fs.open(output_filename + ".h5", 'w') as f:
        model.save(str(output_filename)+'.h5')

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.2, random_state = 42, stratify = y_test)

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
                print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')
                # print(f'Done - penalty: {crit_1}, C: {crit_2}, max_iter: {crit_3}, solver: {crit_4}')
        return best_roc, best_ate, best_params

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()[:10]
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
    Output(rid="ri.vector.main.execute.f5acd343-2d39-4261-82f0-0b454809e43c"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def lr_tlearner(final_data):
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
        # estim_1 = ['elasticnet'] # penalty
        estim_2 = [0, 0.25, 0.5, 0.75, 1] # l1_ratio
        # estim_3 = [100, 500, 1000] # max_iter
        estim_4 = [0.01, 0.1, 1, 10, 100] # C - regularization strength
        # for crit_1 in estim_1:
        for crit_2 in estim_2:
            # for crit_3 in estim_3:
            for crit_4 in estim_4:
                clf = LogisticRegression(penalty='elasticnet', l1_ratio=crit_2, max_iter=100, C=crit_4, solver='saga', class_weight=class_weight_dict)
                clf_learner = BaseTClassifier(learner = clf)
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
                print(f'l1_ratio {crit_2}, C {crit_4}, roc {roc}')
                # print(f'Done - penalty: {crit_1}, C: {crit_2}, max_iter: {crit_3}, solver: {crit_4}')
        return best_roc, best_ate, best_params

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()[:10]
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
    Output(rid="ri.foundry.main.dataset.3cbc3c8c-65b6-4f67-8c4e-40bc4da8bbe8"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def meta_learners_t(final_data):

    def metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts, threshold, model):
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc = roc_auc_score(y_valid, preds)
        ate = ite.mean()
        # preds[preds>threshold] = 1
        # preds[preds<=threshold] = 0
        # print('Accuracy:', accuracy_score(y_valid, preds))
        print(f'T Learner - {model} ATE: {ate}, ROC score: {roc}')
        return roc, ate

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    ingredient_list = main_df.ingredient_concept_id.unique()
    ingredient_pairs = list(combinations(ingredient_list, 2))
    initial_time = datetime.now()
    # ingredient_pairs = [(739138, 703547)]
    threshold = 0.4
    rocs_r = []
    rocs_l = []
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
        print('Class weights dict', class_weight_dict)

        # T-Learner
        modelt1 = RandomForestClassifier(n_estimators = 400, max_depth = 7, class_weight = class_weight_dict)
        learner_t1 = BaseTClassifier(learner = modelt1)
        learner_t1.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_t1.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts, threshold, 'RandomForestClassifier')
        rocs_r.append(roc)
        ates_r.append(ate)

        modelt2 = LogisticRegression(max_iter=1000, class_weight = class_weight_dict)
        learner_t2 = BaseTClassifier(learner = modelt2)
        learner_t2.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_t2.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts, threshold, 'LogisticRegression')
        rocs_l.append(roc)
        ates_l.append(ate)

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    print('Total time taken:',datetime.now() - initial_time)
    print(f'RandomForest: Median {median(rocs_r)}, Mean {mean(rocs_r)}')
    print(f'LogisticRegression: Median {median(rocs_l)}, Mean {mean(rocs_l)}')
    print(f'RandomForest: Median {median(ates_r)}, Mean {mean(ates_r)}, Max {max(ates_r)}, Min {min(ates_r)}')
    print(f'LogisticRegression: Median {median(ates_l)}, Mean {mean(ates_l)}, Max {max(ates_l)}, Min {min(ates_l)}')

    print('ROC R',rocs_r)
    print('ROC L',rocs_l)
    write_text_file(rocs_r, 'rocs_r')
    write_text_file(rocs_l, 'rocs_l')
    write_text_file(ates_r, 'ates_r')
    write_text_file(ates_l, 'ates_l')

    # output = Transforms.get_output()
    # output_fs = output.filesystem()
    # val = []

    # with output_fs.open('roc_r.txt', 'r') as f: 
    #     f.read('roc_r.txt')
    #     val.append(list(f))

    # print(val)
        

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
        estim_1 = [100, 200] # n_estimators
        estim_2 = ['gini', 'entropy', 'log_loss'] # criterion
        estim_3 = [7, 10] # max_depth
        for crit_1 in estim_1:
            for crit_2 in estim_2:
                for crit_3 in estim_3:
                    clf = RandomForestClassifier(n_estimators = crit_1, criterion = crit_2, max_depth = crit_3, class_weight=class_weight_dict)
                    clf_learner = BaseSClassifier(learner = clf)
                    clf_learner.fit(X=X_train, treatment=t_train, y=y_train)
                    ite, yhat_cs, yhat_ts = clf_learner.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
                    roc, ate = metrics(y_valid, t_valid, ite, yhat_cs, yhat_ts)
                    
                    if roc > best_roc:
                        best_ate = ate
                        best_roc = roc
                        # best_params = {'parameters': [('n_estimators', estimator), ('criterion', criterion), ('max_depth', depth)]}
                        best_params = {'n_estimators': crit_1, 'criterion': crit_2, 'max_depth': crit_3, 'penalty':np.nan, 'C':np.nan, 'max_iter':np.nan, 'solver':np.nan}
                    # print(f'Done - n_estimators: {crit_1}, criterion: {crit_2}, max_depth: {crit_3}')
        return best_roc, best_ate, best_params

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    results_df = pd.DataFrame()
    ingredient_list = main_df.ingredient_concept_id.unique()[:5]
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

    

