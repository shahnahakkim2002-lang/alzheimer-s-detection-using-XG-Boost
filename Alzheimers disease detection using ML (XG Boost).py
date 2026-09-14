import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

#import dataset

data = pd.read_csv("/content/alzheimers_disease_data .csv")
print(data)

#sepeerate features and targets

X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

#split training set and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#XG Boost modelling

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

#training model

X_train_processed = X_train.drop('DoctorInCharge', axis=1)
model.fit(X_train_processed, y_train)

#predictions

X_test_processed = X_test.drop('DoctorInCharge', axis=1)
y_pred = model.predict(X_test_processed)

#accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#classification Report

print(classification_report(y_test, y_pred))
