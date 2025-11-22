import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
df = pd.read_excel("shop_sales_data.xlsx")

# Encode categorical columns
label_encoders = {}
for col in ['Month_Name', 'Season', 'Product_Category']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Targets
features = ['Month_Name_Encoded', 'Season_Encoded', 'Product_Category_Encoded', 'Discount_Applied', 'Holiday_Season']
X = df[features]
y_units = df['Units_Sold']
y_revenue = df['Revenue']

# Build Models
model_units = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

model_revenue = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# Train Models
model_units.fit(X, y_units)
model_revenue.fit(X, y_revenue)

# Predict on Training Data
units_pred = model_units.predict(X)
revenue_pred = model_revenue.predict(X)

# Calculate R² Scores
r2_units = r2_score(y_units, units_pred)
r2_revenue = r2_score(y_revenue, revenue_pred)

print("R² Score for Units Sold prediction:", r2_units)
print("R² Score for Revenue prediction:", r2_revenue)

# Visualizations

# Plotting the R² Scores for comparison (Line graph)
plt.figure(figsize=(8, 6))
plt.plot(['Units Sold', 'Revenue'], [r2_units, r2_revenue], marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('R² Score for Units Sold and Revenue Prediction')
plt.xlabel('Prediction Type')
plt.ylabel('R² Score')
plt.ylim([0, 1])
plt.grid(True)
plt.show()

# Feature Importance for Units Sold Prediction (Pie chart)
feature_importances_units = model_units.named_steps['model'].feature_importances_
plt.figure(figsize=(8, 8))
plt.pie(feature_importances_units, labels=features, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Pastel1", len(features)))
plt.title('Feature Importance for Units Sold Prediction')
plt.show()

# Feature Importance for Revenue Prediction (Pie chart)
feature_importances_revenue = model_revenue.named_steps['model'].feature_importances_
plt.figure(figsize=(8, 8))
plt.pie(feature_importances_revenue, labels=features, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Pastel2", len(features)))
plt.title('Feature Importance for Revenue Prediction')
plt.show()

# User Input Prediction: Predict the number of items sold and revenue based on product name and month

def predict_sales_and_revenue(product_name, month_name, discount_applied=0, holiday_season=False):
    # Encoding user input
    if product_name not in label_encoders['Product_Category'].classes_ or month_name not in label_encoders['Month_Name'].classes_:
        return "Invalid product name or month."
    
    product_encoded = label_encoders['Product_Category'].transform([product_name])[0]
    month_encoded = label_encoders['Month_Name'].transform([month_name])[0]
    
    # Create the input feature vector for prediction
    user_input = np.array([[month_encoded,  
                            label_encoders['Season'].transform([df['Season'].mode()[0]])[0],  
                            product_encoded,  
                            discount_applied, 
                            int(holiday_season)  
                           ]])
    
    predicted_units = model_units.predict(user_input)[0]
    
    predicted_revenue = model_revenue.predict(user_input)[0]
    
    return predicted_units, predicted_revenue

product_name = input("Enter Product Name: ")
month_name = input("Enter Month Name: ")
discount_applied = float(input("Enter Discount Applied: "))
holiday_season = input("Is it a Holiday Season? (yes/no): ").lower() == 'yes'

predicted_units, predicted_revenue = predict_sales_and_revenue(product_name, month_name, discount_applied, holiday_season)

print(f"Predicted Units Sold: {predicted_units *10:.2f}")  
print(f"Predicted Revenue: {predicted_revenue * 150:.2f}")

