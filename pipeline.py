import numpy as np
import pandas as pd

from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.inference.nn import CEVAE

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

@transform_pandas(
    Output(rid="ri.vector.main.execute.70189f47-6260-4ec8-942e-c5e0d4bdbd98"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def cevae(final_data):
    df = final_data.toPandas()
    df =  df[df['age_at_covid'].notna()]
    df = df.reset_index(drop=True)

    X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)
    y = df['severity_final']
    t = df['treatment']
    print(1)

    np.random.seed(3)

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights dict', class_weight_dict)

    cevae_model = CEVAE()
    ite = cevae_model.fit_predict(X=X, treatment=t, y=y)
    print(f'ITE: CEVAE - {ite}')

@transform_pandas(
    Output(rid="ri.vector.main.execute.4300054b-4092-4e59-b6a5-9418a642e834"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def meta_learners(final_data):
    df = final_data.toPandas()
    df =  df[df['age_at_covid'].notna()]
    df = df.reset_index(drop=True)

    X = df.drop(['person_id','severity_final', 'ingredient_concept_id', 'treatment'], axis=1)

    y = df['severity_final']
    t = df['treatment']

    np.random.seed(3)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)

    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.7, random_state = 42, stratify = y_test)

    # t_train = t[X_train.index]
    # t_train = t_train.values
    # t_test = t[X_test.index]
    # t_test = t_test.values
    # t_valid = t[X_valid.index]
    # t_valid = t_valid.values

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights dict', class_weight_dict)

    # T-Learner
    modelt1 = RandomForestClassifier(n_estimators = 500, max_depth = 20, class_weight = class_weight_dict)
    learner_t1 = BaseTClassifier(learner = modelt1)
    ate_t1 = learner_t1.estimate_ate(X=X, treatment=t, y=y)
    print(f"ATE T-Learner: RandomForest - Mean {ate_t1[0]}, LB {ate_t1[1]}, UB {ate_t1[2]}")

    modelt2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    learner_t2 = BaseTClassifier(learner = modelt2)
    ate_t2 = learner_t2.estimate_ate(X=X, treatment=t, y=y)
    print(f"ATE T-Learner: Logistic Regression - Mean {ate_t1[0]}, LB {ate_t1[1]}, UB {ate_t1[2]}")

    # S-Learner
    models1 = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight = class_weight_dict)
    learner_s1 = BaseSClassifier(learner = models1)
    ate_s1 = learner_s1.estimate_ate(X=X, treatment=t, y=y)
    print("ATE S-Learner: RandomForest", ate_s1)

    models2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    learner_s2 = BaseSClassifier(learner = models2)
    ate_s2 = learner_s2.estimate_ate(X=X, treatment=t, y=y)
    print("ATE S-Learner: Logistic Regression", ate_s2)

    # X-Learner
    modelx1_c = RandomForestClassifier(n_estimators=500, max_depth=6, class_weight = class_weight_dict)
    modelx1_r = RandomForestRegressor(n_estimators=500, max_depth=6)
    learner_x1 = BaseXClassifier(outcome_learner = modelx1_c, effect_learner = modelx1_r)
    ate_x1 = learner_x1.estimate_ate(X=X, treatment=t, y=y)
    print("ATE X-Learner: RandomForest", ate_x1)

    modelx2_c = LogisticRegression(max_iter=1000000, class_weight = class_weight_dict)
    modelx2_r = LinearRegression()
    learner_x2 = BaseXClassifier(outcome_learner = modelx2_c, effect_learner = modelx2_r)
    ate_x2 = learner_x2.estimate_ate(X=X, treatment=t, y=y)
    print("ATE X-Learner: Logistic Regression", ate_x2)
    

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
    

