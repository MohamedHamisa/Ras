import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to generate realistic data with slight variations
def generate_realistic_data(n=100000):
    regions = ["Cairo New City", "Administrative Capital", "Sheikh Zayed", "Cairo Suburbs", "New Cities"]
    unit_types = ["Residential", "Commercial", "Administrative"]
    unit_states = ["Ready", "Under Construction", "Finishing Stage"]
    years = list(range(2019, 2026))
    data = []
    for _ in range(n):
        region = np.random.choice(regions)
        unit_type = np.random.choice(unit_types)
        unit_state = np.random.choice(unit_states)
        year_built = np.random.choice(years)
        num_units = np.random.randint(1, 10)
        area = np.random.randint(50, 200)
        base_price = 5000 if region == "Cairo Suburbs" else 10000
        # Introduce slight variation in base price
        base_price += np.random.randint(-500, 500)
        price_increase = (year_built - 2019) * 1000
        average_price = base_price + price_increase + (area * 10)
        if unit_type == "Commercial":
            average_price *= 1.5
        data.append([region, unit_type, unit_state, year_built, num_units, area, average_price])
    return pd.DataFrame(data, columns=["Region", "Unit Type", "Unit State", "Year Built", "Number of Units", "Area (m²)", "Average Price per m² (EGP)"])

# Generate data
real_estate_df = generate_realistic_data(n=100000)

# Define numerical and categorical columns
numerical_cols = ['Year Built', 'Number of Units', 'Area (m²)']
categorical_cols = ['Region', 'Unit Type', 'Unit State']

# Set up ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Create a pipeline with preprocessor and model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data
X = real_estate_df.drop(['Average Price per m² (EGP)'], axis=1)
y = real_estate_df['Average Price per m² (EGP)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
cv_scores = cross_val_score(pipeline, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean Cross-Validation R² Score: {cv_scores.mean():.2f}")

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature Importance Analysis
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']
# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_
# Create a DataFrame for feature importances
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importance)

# Prediction loop with error handling
while True:
    print("\nEnter details to predict price per square meter (or type 'كفاية' to exit):")
    region = input("Region (e.g., Cairo New City): ")
    if region.lower() == 'كفاية':
        break
    unit_type = input("Unit Type (Residential, Commercial, Administrative): ")
    unit_state = input("Unit State (Ready, Under Construction, Finishing Stage): ")
    try:
        year_built = int(input("Year Built: "))
        num_units = int(input("Number of Units: "))
        area = float(input("Area (m²): "))
    except ValueError:
        print("Invalid input. Please enter numerical values for Year Built, Number of Units, and Area.")
        continue

    input_data = {
        'Region': [region],
        'Unit Type': [unit_type],
        'Unit State': [unit_state],
        'Year Built': [year_built],
        'Number of Units': [num_units],
        'Area (m²)': [area]
    }

    input_df = pd.DataFrame(input_data)
    try:
        predicted_price = pipeline.predict(input_df)[0]
        print(f"Predicted price per square meter: {predicted_price:.2f} EGP")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
