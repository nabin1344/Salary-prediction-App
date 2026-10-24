import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load the dataset
data = pd.read_csv("Experience-salary.csv")

# 2. Select input (X) and output (y)
X = data[["exp(in months)"]]
y = data["salary(in thousands)"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)



# 6. Save the model
joblib.dump(model, "salary_model.joblib")

print("Model trained and saved!")
