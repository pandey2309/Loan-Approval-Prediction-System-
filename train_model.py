import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load Dataset
# =========================

df = pd.read_csv("dataset/loan_approval_dataset.csv")

print("Dataset Shape:", df.shape)

# =========================
# Feature Engineering
# =========================

df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

# =========================
# Encoding
# =========================

le_gender = LabelEncoder()
le_married = LabelEncoder()
le_education = LabelEncoder()
le_self = LabelEncoder()
le_property = LabelEncoder()
le_target = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Married"] = le_married.fit_transform(df["Married"])
df["Education"] = le_education.fit_transform(df["Education"])
df["Self_Employed"] = le_self.fit_transform(df["Self_Employed"])
df["Property_Area"] = le_property.fit_transform(df["Property_Area"])
df["Loan_Status"] = le_target.fit_transform(df["Loan_Status"])

# =========================
# Features & Target
# =========================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Models
# =========================

# =========================
# Models Comparison
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

print("\nModel Results\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy*100:.2f}%")

# =========================
# Final Model Selection
# =========================

best_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

best_model.fit(X_train, y_train)

rf_predictions = best_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nFinal Selected Model: Random Forest")
print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")

# =========================
# Save Model
# =========================

joblib.dump(
    best_model,
    "models/loan_model.pkl"
)

print("\nModel Saved Successfully!")
print("Saved Model: Random Forest")