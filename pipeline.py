import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

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
    
    if metric == 'roc':
        filename = 'roc.txt'
    else:
        filename = 'ate.txt'
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
    Output(rid="ri.vector.main.execute.4300054b-4092-4e59-b6a5-9418a642e834"),
    final_data=Input(rid="ri.foundry.main.dataset.189cbacb-e1b1-4ba8-8bee-9d6ee805f498")
)
def meta_learners(final_data):

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
        y = df['severity_final'].values
        t = df['treatment'].values

        np.random.seed(0)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)

        # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.7, random_state = 42, stratify = y_test)

        # y_train, y_valid, y_test = y_train.values.reshape(-1,1), y_valid.values.reshape(-1,1), y_test.values.reshape(-1,1)
        # t_train = t[X_train.index]
        # t_train = t_train.values.reshape(-1,1)
        # t_test = t[X_test.index]
        # t_test = t_test.values.reshape(-1,1)
        # t_valid = t[X_valid.index]
        # t_valid = t_valid.values.reshape(-1,1)

        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)
        class_weight_dict = dict(enumerate(class_weights))
        print('Class weights dict', class_weight_dict)

        # T-Learner
        modelt1 = RandomForestClassifier(n_estimators = 500, max_depth = 20, class_weight = class_weight_dict)
        # modelt1.fit(X=X_train, y=np.concatenate([y_train, t_train], 1))
        # modelt1_preds = modelt1.predict_proba(X_valid)
        # y0_pred = modelt1_preds[:, 0]
        # y1_pred = modelt1_preds[:, 1]

        # print('y0_pred',y0_pred)
        # print('y1_pred',y1_pred)
        # print('t_valid',t_valid)
        # preds = (1. - t_valid) * y0_pred + t_valid * y1_pred
        # print('preds',preds)
        # print('modelt1_preds',modelt1_preds)
        learner_t1 = BaseTClassifier(learner = modelt1)
        pred_t1 = learner_t1.fit_predict(X=X, treatment=t, y=y, verbose=True)
        print('pred_t1:',pred_t1)
        ate_t1 = learner_t1.estimate_ate(X=X, treatment=t, y=y)
        print(f"ATE T-Learner: RandomForest - Mean {ate_t1[0]}, LB {ate_t1[1]}, UB {ate_t1[2]}")

        modelt2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
        learner_t2 = BaseTClassifier(learner = modelt2)
        ate_t2 = learner_t2.estimate_ate(X=X, treatment=t, y=y)
        print(f"ATE T-Learner: Logistic Regression - Mean {ate_t2[0]}, LB {ate_t2[1]}, UB {ate_t2[2]}")

        # S-Learner
        models1 = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight = class_weight_dict)
        learner_s1 = BaseSClassifier(learner = models1)
        ate_s1 = learner_s1.estimate_ate(X=X, treatment=t, y=y)
        print("ATE S-Learner: RandomForest", ate_s1)

        models2 = LogisticRegression(max_iter=10000, class_weight = class_weight_dict)
        learner_s2 = BaseSClassifier(learner = models2)
        ate_s2 = learner_s2.estimate_ate(X=X, treatment=t, y=y)
        print("ATE S-Learner: Logistic Regression", ate_s2)

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
    

