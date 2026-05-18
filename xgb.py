import pandas as pd
from preprocessing import preprocessing, prepare_data
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from xgboost import plot_importance
import optuna
from sklearn.metrics import f1_score

def objective(trial, x_train, Y_train, x_val, Y_val):    
    param = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 15),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'early_stopping_rounds': 30,
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.5,1.0),
        'min_child_weight': trial.suggest_int('min_child_weight',1,5),
        'gamma': trial.suggest_float('gamma',0,5),
    }

    model = XGBClassifier(**param, random_state=42, eval_metric='logloss')
    model.fit(x_train, Y_train, eval_set=[(x_val, Y_val)], verbose=False)     
      
    preds = model.predict_proba(x_val)[:,1]
    score = f1_score(Y_val, (preds >= 0.3).astype(int))
    
    return score

dataset = pd.read_csv('healthcare_fraud_detection.csv')
df = preprocessing(dataset)

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(df)

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_train, Y_train, X_val, Y_val), n_trials=50)

best_params = study.best_params

xgb_model = XGBClassifier(
    ** best_params,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)

train_probs = xgb_model.predict_proba(X_train)[:, 1] 
test_probs = xgb_model.predict_proba(X_test)[:, 1]

threshold = 0.3

train_pred_30 = (train_probs >= threshold).astype(int)
test_pred_30 = (test_probs >= threshold).astype(int)

score = f1_score(Y_test, test_pred_30)

print("Results on training set:")
print(classification_report(Y_train, train_pred_30))

print("Results on test set:")
print(classification_report(Y_test, test_pred_30))

cm = confusion_matrix(Y_test, test_pred_30)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

PrecisionRecallDisplay.from_estimator(xgb_model, X_test, Y_test)
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=10, ax=plt.gca())
plt.tight_layout()
plt.show()

