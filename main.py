import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv("music_data.csv")

# Step 2: Data Preprocessing
# Check for missing values and fill them if necessary
df.fillna(df.mean(), inplace=True)

# Convert categorical labels (genre) into numerical values
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre'])

# Step 3: Feature Selection
X = df.drop(columns=['genre'])  # Features
y = df['genre']  # Target variable

# Step 4: Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Model Training with Hyperparameter Tuning
# Using GridSearchCV to find the best parameters
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Step 7: Evaluate the Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Function to Predict Genre for New Songs
def predict_genre(features):
    """
    Predict the genre based on input features.
    :param features: List of song features [bpm, energy, danceability, loudness, valence]
    :return: Predicted genre
    """
    features_scaled = scaler.transform([features])  # Apply scaling
    genre_index = best_model.predict(features_scaled)[0]
    return label_encoder.inverse_transform([genre_index])[0]

# Example Usage with User Input
print("\nEnter song features for prediction:")
bpm = float(input("Beats Per Minute (BPM): "))
energy = float(input("Energy Level (0-100): "))
danceability = float(input("Danceability (0-100): "))
loudness = float(input("Loudness (dB): "))
valence = float(input("Valence (mood positivity 0-100): "))

new_song_features = [bpm, energy, danceability, loudness, valence]
predicted_genre = predict_genre(new_song_features)
print(f"\nPredicted Genre: {predicted_genre}")
