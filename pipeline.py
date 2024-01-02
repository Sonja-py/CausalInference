import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from copy import deepcopy

from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.inference.nn import CEVAE

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

# Save error metrics
def write_text_file(data, metric):
    output = Transforms.get_output()
    output_fs = output.filesystem()
    metric, learner = metric.split('_')
    
    if metric == 'roc' and learner == 's':
        filename = 'roc_s.txt'
    elif metric == 'roc' and learner == 't':
        filename = 'roc_t.txt'
    elif metric == 'ate' and learner == 's':
        filename = 'ate_s.txt'
    else:
        filename = 'ate_t.txt'
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

        np.random.seed(3)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.2, random_state = 42, stratify = y_test)

        t_train = t[X_train.index]
        t_test = t[X_test.index]
        t_valid = t[X_valid.index]

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        print('Class weights dict', class_weight_dict)

        cevae_model = CEVAE(num_epochs = 10, batch_size = 128, learning_rate = 1e-2, num_samples = 100)
        loss = cevae_model.fit(X=X_train, treatment=t_train, y=y_train)
        print('Loss:',loss)
        # plt.plot(loss)
        # plt.title(f'CEVAE loss for combo {combination}')
        # plt.xlabel('Epoch')
        # plt.ylabel('Training Loss')
        # plt.show()

        ite = cevae_model.predict(X_valid.to_numpy())
        ate = ite.mean()
        print('ATE:',ate)
        ates.append(ate)
        # save_model(cevae_model, str(combination[0]) + '_' + str(combination[1]))
        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')
        

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c55c9d7f-ff97-450b-8d2c-fabecfaaa8ab"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def meta_learner_s(final_data):

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    ingredient_list = main_df.ingredient_concept_id.unique()[:10]
    ingredient_pairs = list(combinations(ingredient_list, 2))
    rocs_s = []
    ates_s = []

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

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        print('Class weights dict', class_weight_dict)
    
        # S-Learner
        models1 = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight = class_weight_dict)
        learner_s1 = BaseSClassifier(learner = models1)
        learner_s1.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_s1.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc_score = roc_auc_score(y_valid, preds)
        print('S Learner - RandomForest ATE:',ite.mean())
        print('S Learner - RandomForest ROC score:', roc_score)
        rocs_s.append(roc_score)
        ates_s.append(ite.mean())

        models2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
        learner_s2 = BaseSClassifier(learner = models2)
        learner_s2.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_s2.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc_score = roc_auc_score(y_valid, preds)
        print('S Learner - LogisticRegression ATE:',ite.mean())
        print('S Learner - LogisticRegression ROC score:', roc_score)
        rocs_s.append(roc_score)
        ates_s.append(ite.mean())

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

    print(f'Median {np.array(roc_s).median()}, Mean {np.array(roc_s).mean()}')
    write_text_file(rocs_s, 'roc_s')
    write_text_file(ates_s, 'ate_s')

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

    # Create and get the data for pair of different antidepressants
    main_df = final_data.toPandas()
    ingredient_list = main_df.ingredient_concept_id.unique()[:10]
    ingredient_pairs = list(combinations(ingredient_list, 2))
    rocs_t = []
    ates_t = []

    for idx, combination in enumerate(ingredient_pairs):
        start_time = datetime.now()
        print(f'-----------Running Meta-Learners for drug pair: {combination}. It is number {idx+1} of {len(ingredient_pairs)} -----------')
        df = main_df[main_df.ingredient_concept_id.isin(list(combination))]
        df['treatment'] = df['ingredient_concept_id'].apply(lambda x: 0 if x == combination[0] else 1)

        X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
        y = df['severity_final']
        t = df['treatment']

        np.random.seed(3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)
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
        modelt1 = RandomForestClassifier(n_estimators = 100, max_depth = 15, class_weight = class_weight_dict)
        learner_t1 = BaseTClassifier(learner = modelt1)
        learner_t1.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_t1.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc_score = roc_auc_score(y_valid, preds)
        print('T Learner - RandomForest ATE:',ite.mean())
        print('T Learner - RandomForest ROC score:', roc_score)
        rocs_t.append(roc_score)
        ates_t.append(ite.mean())

        modelt2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
        learner_t2 = BaseTClassifier(learner = modelt2)
        learner_t1.fit(X=X_train, treatment=t_train, y=y_train)
        ite, yhat_cs, yhat_ts = learner_t1.predict(X=X_valid, treatment=t_valid, y=y_valid, return_components=True, verbose=True)
        yhat_cs, yhat_ts = np.array(list(yhat_cs.values())[0]), np.array(list(yhat_ts.values())[0])
        preds = (1. - t_valid) * yhat_cs + t_valid * yhat_ts
        roc_score = roc_auc_score(y_valid, preds)
        print('T Learner - LogisticRegression ATE:',ite.mean())
        print('T Learner - LogisticRegression ROC score:', roc_score)
        rocs_t.append(roc_score)
        ates_t.append(ite.mean())

        print(f'Time taken for combination {idx+1} is {datetime.now() - start_time}')

        # # X-Learner
        # modelx1_c = RandomForestClassifier(n_estimators=500, max_depth=6, class_weight = class_weight_dict)
        # modelx1_r = RandomForestRegressor(n_estimators=500, max_depth=6)
        # learner_x1 = BaseXClassifier(outcome_learner = modelx1_c, effect_learner = modelx1_r)
        # ate_x1 = learner_x1.estimate_ate(X=X, treatment=t, y=y)
        # print("ATE X-Learner: RandomForest", ate_x1)

        # modelx2_c = LogisticRegression(max_iter=1000000, class_weight = class_weight_dict)
        # modelx2_r = LinearRegression()
        # learner_x2 = BaseXClassifier(outcome_learner = modelx2_c, effect_learner = modelx2_r)
        # ate_x2 = learner_x2.estimate_ate(X=X, treatment=t, y=y)
        # print("ATE X-Learner: Logistic Regression", ate_x2)

    print(f'Median {np.array(roc_t).median()}, Mean {np.array(roc_t).mean()}')
    write_text_file(rocs_t, 'roc_t')
    write_text_file(ates_t, 'ate_t')
        

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
        

