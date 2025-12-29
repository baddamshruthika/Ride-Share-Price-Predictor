import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Simulate a Dataset (In a real scenario, we would load 'uber_fares.csv')
# We create dummy data here so the script runs immediately without external files.
print("Generating dummy dataset...")
np.random.seed(42)
data_size = 1000

data = {
    'distance_km': np.random.uniform(1, 50, data_size),
    'passenger_count': np.random.randint(1, 6, data_size),
    'hour_of_day': np.random.randint(0, 24, data_size),
    # Base fare + distance cost + peak hour surge logic
    'fare_amount': np.zeros(data_size) 
}

df = pd.DataFrame(data)

# Simple logic to determine price (Ground Truth)
# Price = Base($5) + ($2 * distance) + (surge if hour is 17-19)
def calculate_fare(row):
    price = 5.0 + (2.0 * row['distance_km'])
    if 17 <= row['hour_of_day'] <= 19: # Peak hours
        price *= 1.5
    return price

df['fare_amount'] = df.apply(calculate_fare, axis=1)

# 2. Data Preprocessing
print("Preprocessing data...")
# Features (X) and Target (y)
X = df[['distance_km', 'passenger_count', 'hour_of_day']]
y = df['fare_amount']

# Split into Train and Test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Model Training Complete.")
print(f"Accuracy (R2 Score): {accuracy * 100:.2f}%")
print("-" * 30)

# Example Prediction
print("\nExample Prediction:")
sample_ride = [[12.5, 2, 18]] # 12.5 km, 2 passengers, 6 PM (Peak hour)
predicted_price = model.predict(sample_ride)
print(f"Predicted Fare for 12.5km ride at 6 PM: ${predicted_price[0]:.2f}")
