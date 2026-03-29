import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("placementdata.csv")

# Convert target column to numeric
df['PlacementStatus'] = df['PlacementStatus'].map({'Placed':1,'NotPlaced':0})

# Features and target
X = df[['CGPA','Internships','Projects','AptitudeTestScore','SoftSkillsRating']]
y = df['PlacementStatus']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML Pipeline (Scaling + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
pickle.dump(pipeline, open("placement_model.pkl", "wb"))

print("\nModel trained and saved successfully!")