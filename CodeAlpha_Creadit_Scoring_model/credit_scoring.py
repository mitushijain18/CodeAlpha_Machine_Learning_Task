import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("credit_data.csv")

# Features and target
X = data[['Income', 'Debt', 'PaymentHistory', 'LoanAmount']]
y = data['Creditworthy']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("===================================")
print("      CREDIT SCORING MODEL")
print("===================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nPredictions:")
print(predictions)

print("\nActual Values:")
print(y_test.values)