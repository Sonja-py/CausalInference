import numpy as np
import pandas as pd

# from causalml.inference.meta import BaseSLearner, BaseSClassifier, BaseTClassifier
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

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
    treatment = df['treatment']

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights dict', class_weight_dict)

    modelt1 = RandomForestClassifier(n_estimators = 1000, max_depth = 20, class_weight = class_weight_dict)
    learner_t1 = BaseTClassifier(learner = modelt1)
    ate_t1 = learner_t1.estimate_ate(X=X, treatment=treatment, y=y)
    print("ATE T-Learner: RandomForest", ate_t1)

    modelt2 = LogisticRegression(max_iter=100000, class_weight = class_weight_dict)
    learner_t2 = BaseTClassifier(learner = modelt2)
    ate_t2 = learner_t2.estimate_ate(X=X, treatment=treatment, y=y)
    print("ATE T-Learner: Logistic Regression", ate_t2)

    models1 = RandomForestClassifier(n_estimators=1000, max_depth=20, class_weight = class_weight_dict)
    learner_s1 = BaseSClassifier(learner = models1)
    ate_s1 = learner_s1.estimate_ate(X=X, treatment=treatment, y=y)
    print("ATE S-Learner: RandomForest", ate_s1)

    models2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    learner_s2 = BaseSClassifier(learner = models2)
    ate_s2 = learner_s2.estimate_ate(X=X, treatment=treatment, y=y)
    print("ATE S-Learner: Logistic Regression", ate_s2)

    # modelx1_c = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight = class_weight_dict)
    # modelx1_r = RandomForestRegressor(n_estimators=100, max_depth=6, class_weight = class_weight_dict)
    # learner_x1 = BaseXClassifier(outcome_learner = modelx1_c, effect_learner = modelx1_r)
    # ate_x1 = learner_x1.estimate_ate(X=X, treatment=treatment, y=y)
    # print("ATE X-Learner: RandomForest", ate_x1)

    # modelx2_c = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
    # modelx2_r = LinearRegression(class_weight = class_weight_dict)
    # learner_x2 = BaseXClassifier(outcome_learner = modelx2_c, effect_learner = modelx2_r)
    # ate_x2 = learner_x2.estimate_ate(X=X, treatment=treatment, y=y)
    # print("ATE X-Learner: Logistic Regression", ate_x2)
    

