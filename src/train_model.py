import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ✅ Load with error handling
df = pd.read_csv(
    "data/resume_data.csv",
    encoding="utf-8-sig",
    on_bad_lines="skip"   # 🔥 SKIP BROKEN ROWS
)

# Clean columns
df.columns = df.columns.str.strip()

# Fill missing
df = df.fillna("")

# Clean list-like text
def clean_list(text):
    return re.sub(r"[\[\]'\n]", " ", str(text))

if "skills" in df.columns:
    df["skills"] = df["skills"].apply(clean_list)

if "responsibilities" in df.columns:
    df["responsibilities"] = df["responsibilities"].apply(clean_list)

# Combine text safely
df["combined_text"] = (
    df.get("career_objective", "") + " " +
    df.get("skills", "") + " " +
    df.get("responsibilities", "")
)

# Detect target column
target_col = None
for col in df.columns:
    if "job_position" in col:
        target_col = col
        break

if target_col is None:
    raise Exception("❌ Target column not found")

# Features
X = df["combined_text"]
y = df[target_col]

# Remove empty rows
mask = (X.str.strip() != "") & (y.str.strip() != "")
X = X[mask]
y = y[mask]

print("Training samples:", len(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)


from sklearn.metrics import accuracy_score

# Evaluate
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Save
import os

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")