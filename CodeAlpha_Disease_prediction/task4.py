
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

np.random.seed(42)
num_patients = 600

data = {
    'Pregnancies': np.random.randint(0, 10, size=num_patients),
    'Glucose': np.random.randint(70, 200, size=num_patients),
    'BloodPressure': np.random.randint(60, 110, size=num_patients),
    'SkinThickness': np.random.randint(10, 50, size=num_patients),
    'Insulin': np.random.randint(15, 300, size=num_patients),
    'BMI': np.random.uniform(18.5, 45.0, size=num_patients),
    'DiabetesPedigreeFunction': np.random.uniform(0.1, 1.5, size=num_patients),
    'Age': np.random.randint(21, 70, size=num_patients)
}

df = pd.DataFrame(data)


risk_score = (df['Glucose'] * 0.04) + (df['BMI'] * 0.07) + (df['Age'] * 0.02) + (df['DiabetesPedigreeFunction'] * 0.5)
probability = 1 / (1 + np.exp(-(risk_score - 10.5))) # Sigmoid function mapping
df['Outcome'] = (probability > np.random.uniform(0, 1, size=num_patients)).astype(int)

print("--- Dataset Sample Ingested ---")
print(df.head())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train) # Tree architectures do not strictly require scaling
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("\n=========================================")
print("      LOGISTIC REGRESSION METRICS        ")
print("=========================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob_lr):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

print("\n=========================================")
print("        RANDOM FOREST METRICS            ")
print("=========================================")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob_rf):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.2f})', color='teal')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})', color='crimson')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()

plt.subplot(1, 2, 2)
importances = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='royalblue')
plt.xlabel('Relative Importance Value')
plt.title('Feature Importance Matrix')

plt.tight_layout()
plt.savefig('disease_prediction_metrics.png', dpi=300)
print("\n[INFO] Performance analytics charts saved successfully as 'disease_prediction_metrics.png'")