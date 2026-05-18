import pandas as pd
from preprocessing import preprocessing, prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

dataset = pd.read_csv('healthcare_fraud_detection.csv')

df = preprocessing(dataset)

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(df)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, Y_train)

train_probs = random_forest.predict_proba(X_train)[:, 1]
test_probs = random_forest.predict_proba(X_test)[:, 1]

threshold = 0.3

train_pred = (train_probs >= threshold).astype(int)
test_pred = (test_probs >= threshold).astype(int)

print("Results on training set:")
print(classification_report(Y_train, train_pred))

print("Results on test set:")
print(classification_report(Y_test, test_pred))

cm = confusion_matrix(Y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

PrecisionRecallDisplay.from_estimator(
    random_forest,
    X_test,
    Y_test
)
plt.show()

importances = pd.Series(
    random_forest.feature_importances_,
    index=X_train.columns
)

top_10_features = (
    importances
    .sort_values(ascending=True)
    .tail(10)
)

plt.figure(figsize=(10, 8))
top_10_features.plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()