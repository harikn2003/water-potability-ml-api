import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
df = pd.read_csv("water_potability.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

X = df.drop("Potability", axis=1)
y = df["Potability"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (important for ML quality)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base models
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier()

# ENSEMBLE MODEL 
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1]))

# Save model + scaler
with open("model/model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model + Scaler saved!")