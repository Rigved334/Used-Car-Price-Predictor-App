import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

cars = pd.read_csv(r"C:\Users\Rigved Bhondve\OneDrive\Desktop\ML\Car Price Predictor\data\raw\used_cars.csv")

for i in range(cars.shape[0]):
    price_l = cars.loc[i, "price"].split("$")[1].split(",")
    price = "".join(price_l)
    cars.loc[i, "price"] = int(price)

for i in range(cars.shape[0]):
    milage_l = cars.loc[i, "milage"].split(" ")[0].split(",")
    milage = "".join(milage_l)
    cars.loc[i, "milage"] = int(milage)

cars["price"] = cars["price"].astype(int)
cars["milage"] = cars["milage"].astype(int)

X = cars.drop(columns="price")
y = cars["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# num_imputer = SimpleImputer(strategy="mean")
num_columns = X_train.select_dtypes(include=[np.number]).columns
# num_imputer.fit(X_train_num)

cat_columns = X_train.select_dtypes(exclude=[np.number]).columns
# X_train_cat = X_train_cat.fillna("Missing")

# cat_encoder = OneHotEncoder(sparse_output=False)
# cars_cat_1hot = cat_encoder.fit_transform(X_train_cat)

# scaler = StandardScaler()
# X_train_num_scaled = scaler.fit_transform(X_train_num)

num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy="constant", fill_value="Missing"), OneHotEncoder(handle_unknown="ignore", sparse_output=False))

preprocessing = ColumnTransformer([("num", num_pipeline, num_columns),
                                   ("cat", cat_pipeline, cat_columns)])

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(X_train, y_train)

new_car = pd.DataFrame({
    "brand": ["BMW"],
    "model": ["X5"],
    "model_year": [2022],
    "milage": [18000],
    "fuel_type": ["Gasoline"],
    "engine": ["3.0L I6 Turbo"],
    "transmission": ["Automatic"],
    "ext_col": ["Black"],
    "int_col": ["Brown"],
    "accident": ["None reported"],
    "clean_title": ["Yes"]
})

predicted_price = lin_reg.predict(new_car)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")

joblib.dump(lin_reg, "Used_car_price.pkl")