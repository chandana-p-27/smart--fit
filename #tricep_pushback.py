#tricep_pushback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

df = pd.read_csv("tricep_pushback_dataset.csv")

df["label"] = df["label"].apply(lambda x: "correct" if x == "correct" else "incorrect")

df["elbow_velocity"] = df["elbow_angle"].diff().fillna(0)
df["shoulder_movement"] = df["shoulder_angle"].diff().abs().fillna(0)
df["elbow_acceleration"] = df["elbow_velocity"].diff().fillna(0)

df_majority = df[df.label == "incorrect"]
df_minority = df[df.label == "correct"]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[[
    "elbow_angle",
    "shoulder_angle",
    "hip_angle",
    "elbow_velocity",
    "shoulder_movement",
    "elbow_acceleration"
]]

y = df_balanced["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))