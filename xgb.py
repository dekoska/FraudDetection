import pandas as pd
from preprocessing import preprocessing, prepare_data
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from xgboost import plot_importance

dataset = pd.read_csv('healthcare_fraud_detection.csv')

df = preprocessing(dataset)

x_train, x_test, Y_train, Y_test = prepare_data(df)

xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1,    
    max_depth=6,          
    scale_pos_weight=11,  
    random_state=42,
    eval_metric='logloss' 
)

xgb_model.fit(x_train, Y_train)

train_probs = xgb_model.predict_proba(x_train)[:, 1] 
test_probs = xgb_model.predict_proba(x_test)[:, 1]

threshold = 0.3

train_pred_30 = (train_probs >= threshold).astype(int)
test_pred_30 = (test_probs >= threshold).astype(int)

print("Results on training set:")
print(classification_report(Y_train, train_pred_30))

print("Results on test set:")
print(classification_report(Y_test, test_pred_30))

cm = confusion_matrix(Y_test, test_pred_30)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

PrecisionRecallDisplay.from_estimator(xgb_model, x_test, Y_test)
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=10, ax=plt.gca())
plt.tight_layout()
plt.show()