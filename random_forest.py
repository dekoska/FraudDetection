import pandas as pd
from preprocessing import preprocessing, prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

dataset = pd.read_csv('healthcare_fraud_detection.csv')

df = preprocessing(dataset)
x_train, x_test, Y_train, Y_test = prepare_data(df)

random_forest = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        min_samples_leaf=5, 
        random_state=42, 
        class_weight='balanced' 
    ) 

random_forest.fit(x_train, Y_train)

train_probs = random_forest.predict_proba(x_train)[:, 1] 
test_probs = random_forest.predict_proba(x_test)[:, 1]

threshold = 0.3

train_pred_30 = (train_probs >= threshold).astype(int)
test_pred_30 = (test_probs >= threshold).astype(int)

print("Results on training set:")
print(classification_report(Y_train, train_pred_30))

print("Results on test set:")
print(classification_report(Y_test, test_pred_30))

importances = pd.Series(random_forest.feature_importances_, index=x_train.columns)
top_10_features = importances.sort_values(ascending=True).tail(10)

plt.figure(figsize=(10, 8))
top_10_features.plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
