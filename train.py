import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv("data/scoring.csv")

X = data.drop(columns=["default"]).values
y = data["default"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print(f"Отказано: {y_pred.mean() * 100:.0f}%")
# print(f"Точность: {precision * 100:.0f}%")
# print(f"Полнота: {recall * 100:.0f}%")

joblib.dump(model, "model.pkl")