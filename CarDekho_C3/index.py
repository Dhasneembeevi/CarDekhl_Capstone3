import ast
import pandas as pd
import streamlit as st
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

st.title("Car Dheko - Used Car Price Prediction")


def extract_feature_from_top(data, search_key, default=None):
    """Extract specific feature from the 'top' list of dictionaries."""
    if isinstance(data, dict) and "top" in data:
        for item in data["top"]:
            key = item.get("key")
            value = item.get("value")
            if key == search_key:
                return value
    return default


def parse_json(data):
    """Safely parse data, handling both dict and string cases."""
    if isinstance(data, dict):
        return data
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return {}


def extract_feature(data, key, default=None):
    """Extract specific feature from JSON-like data."""
    if isinstance(data, dict):
        return data.get(key, default)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("key") == key:
                return item.get("value", default)
    return default


downloads_folder = os.path.expanduser("~/Downloads")
xlsx_files = [
    "kolkata_cars.xlsx",
    "jaipur_cars.xlsx",
    "hyderabad_cars.xlsx",
    "delhi_cars.xlsx",
    "chennai_cars.xlsx",
    "bangalore_cars.xlsx",
]

all_features_df = pd.DataFrame()

for file in xlsx_files:
    file_path = os.path.join(downloads_folder, file)
    df = pd.read_excel(file_path)

    if "new_car_detail" in df.columns:
        df["new_car_detail"] = df["new_car_detail"].apply(parse_json)

        df_feature = pd.DataFrame(
            {
                "Price": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "price")
                ),
                "Transmission": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "transmission")
                ),
                "Body Type": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "bt")
                ),
                "Fuel Type": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "ft")
                ),
                "Kilometers Driven": df["new_car_detail"].apply(
                    lambda x: (
                        extract_feature(x, "km").replace(",", "")
                        if extract_feature(x, "km")
                        else "0"
                    )
                ),
                "Model Year": df["new_car_detail"].apply(
                    lambda x: (
                        str(extract_feature(x, "modelYear")).replace(",", "").strip()
                        if extract_feature(x, "modelYear") is not None
                        else "0"
                    )
                ),
                "Previous Owners": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "ownerNo")
                ),
                "Model Year": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "modelYear")
                ),
                "OEM": df["new_car_detail"].apply(lambda x: extract_feature(x, "oem")),
                "Model": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "model")
                ),
                "Variant Name": df["new_car_detail"].apply(
                    lambda x: extract_feature(x, "variantName")
                ),
                # "Trending Text": df["new_car_detail"].apply(
                #     lambda x: extract_feature(x, "trendingText", {}).get("heading", "")
                # ),
                "City": file.split("_")[0],
            }
        )

        def clean_year(value):
            if pd.isna(value):
                return "Unknown"
            return str(int(value))

        all_features_df = pd.concat([all_features_df, df_feature])
        all_features_df["Model Year"] = df["new_car_detail"].apply(
            lambda x: clean_year(extract_feature(x, "modelYear"))
        )
        
        

        # Define a function to convert 'Price' values
        def convert_price(value):
            # Check if the value is a string before processing
            if isinstance(value, str):
                value = value.strip()
                if '₹' in value:
                    value = value.replace('₹', '').replace(',', '')  # Remove currency symbol and commas
                    if 'Lakh' in value:
                        value = float(value.replace('Lakh', '').strip()) * 100000
                    elif 'Crore' in value:
                        value = float(value.replace('Crore', '').strip()) * 10000000
                    else:
                        value = float(value)
                return value
            # If value is not a string, return as is
            return value

        # Example usage
        all_features_df['Price'] = all_features_df['Price'].apply(convert_price)

        # Check the cleaned data
        print(all_features_df['Price'].head())





    else:
        st.write(f"The file {file} does not contain the 'new_car_detail' column.")

if "new_car_overview" in df.columns:

    df["new_car_overview"] = df["new_car_overview"].apply(parse_json)

    df_feature_overview = pd.DataFrame(
        {
            "Registration Year": df["new_car_overview"].apply(
                lambda x: extract_feature_from_top(x, "Registration Year")
            ),
            "Insurance Validity": df["new_car_overview"].apply(
                lambda x: extract_feature_from_top(x, "Insurance Validity")
            ),
            # "Fuel Type Overview": df["new_car_overview"].apply(
            #     lambda x: extract_feature_from_top(x, "Fuel Type")
            # ),
            "Seats": df["new_car_overview"].apply(
                lambda x: extract_feature_from_top(x, "Seats")
            ),
            # "Ownership": df["new_car_overview"].apply(
            #     lambda x: extract_feature_from_top(x, "Ownership")
            # ),
            "Year of Manufacture": df["new_car_overview"].apply(
                lambda x: extract_feature_from_top(x, "Year of Manufacture")
            ),
        }
    )

    all_features_df = pd.concat([all_features_df, df_feature_overview], axis=1)

    def clean_year(value):
        if pd.isna(value):
            return "Unknown"
        return str(int(value))

    all_features_df["Year of Manufacture"] = df["new_car_overview"].apply(
        lambda x: clean_year(extract_feature_from_top(x, "Year of Manufacture"))
    )

    def extract_year(value):

        if isinstance(value, dict):
            value = value.get("text", "")

        if isinstance(value, str):

            match = re.search(r"\b(\d{4})\b", value)
            if match:
                return match.group(1)
        return "Unknown"

    all_features_df["Registration Year"] = df["new_car_overview"].apply(
        lambda x: extract_year(extract_feature_from_top(x, "Registration Year"))
    )

    # print(all_features_df["Registration Year"].head())


else:
    st.write("The DataFrame does not contain the 'new_car_overview' column.")

# st.write(all_features_df)
# print(all_features_df)
all_features_df.dropna(axis=0, inplace=True)
# print(all_features_df.isna().sum())
print(all_features_df)
print(all_features_df.columns)




unique_values = all_features_df["Transmission"].unique()
unique_values_bt = all_features_df["Body Type"].unique()

filtered_body_types = [body for body in unique_values_bt if body]

unique_values_ft = all_features_df["Fuel Type"].unique()
unique_values_my = all_features_df["Model Year"].unique()
unique_values_my_sorted = sorted(unique_values_my, key=lambda x: int(x))


unique_values_oem = all_features_df["OEM"].unique()
unique_values_ml = all_features_df["Model"].unique()
unique_values_city = all_features_df["City"].unique()
unique_values_city_capitalized = [city.capitalize() for city in unique_values_city]
unique_values_vn = all_features_df["Variant Name"].unique()
unique_values_iv = all_features_df["Insurance Validity"].unique()
unique_values_seats = all_features_df["Seats"].unique()

# all_features_df = pd.get_dummies(all_features_df, columns=['Fuel Type', 'Transmission', 'Body Type','OEM','Variant Name'])
# print(all_features_df.head())

categorical_columns = ['Transmission', 'Body Type', 'Fuel Type', 'OEM', 'Model', 'Variant Name', 'City','Insurance Validity']
all_features_df = pd.get_dummies(all_features_df, columns=categorical_columns)
print(all_features_df.head())

with st.sidebar:
    st.header("Car Features")
    city = st.selectbox("City", unique_values_city_capitalized, key="city")
    fuel_type = st.selectbox("Fuel Type", unique_values_ft, key="fuel_type")
    body_type = st.selectbox("Body Type", filtered_body_types, key="body_type")
    variant_name = st.selectbox("Variant Name", unique_values_vn, key="variant_name")
    model_year = st.selectbox(
        "Year of Manufacture", unique_values_my_sorted, key="model_year"
    )
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    transmission = st.selectbox("Transmission", unique_values, key="transmission")
    owner_no = st.number_input(
        "Number of Previous Owners", min_value=0, max_value=5, step=1
    )
    manufacturer = st.selectbox(
        "Original Equipment Manufacturer (OEM)", unique_values_oem, key="manufacturer"
    )
    car_model = st.selectbox("Car Model", unique_values_ml, key="car_model")
    insurance = st.selectbox("Insurance Validity", unique_values_iv, key="insurance")
    seats = st.selectbox("Seats", unique_values_seats, key="seats")

    st.button("Estimate Price")


st.write("## Selected Car Details")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**City**: {city}")
    st.write(f"**Fuel Type**: {fuel_type}")
    st.write(f"**Body Type**: {body_type}")
    st.write(f"**Variant Name**: {variant_name}")
    st.write(f"**Kilometers Driven**: {kilometers_driven}")
    st.write(f"**Transmission**: {transmission}")

with col2:
    st.write(f"**Previous Owners**: {owner_no}")
    st.write(f"**Manufacturer**: {manufacturer}")
    st.write(f"**Model**: {car_model}")
    st.write(f"**Model Year**: {model_year}")
    st.write(f"**Insurance Validity**: {insurance}")
    st.write(f"**Seats**: {seats}")


numerical_features = ['Kilometers Driven', 'Price']
scaler = MinMaxScaler()
all_features_df[numerical_features] = scaler.fit_transform(all_features_df[numerical_features])

print(all_features_df.head())


sns.boxplot(x='Price', data=all_features_df)
print(plt.show())

#Split Data:

X = all_features_df.drop(['Price'], axis=1)  # Features
y = all_features_df['Price']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Selection

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Hyperparameter Tuning:


param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

#Model Evaluation

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

if st.button("Predict"):
    prediction = model.predict([[city, fuel_type, body_type, model_year]])
    st.write(f"The predicted price is ₹{prediction[0]:.2f} Lakh")


with st.sidebar:
    st.header("Car Features")
    city = st.selectbox("City", unique_values_city_capitalized, key="city")
    fuel_type = st.selectbox("Fuel Type", unique_values_ft, key="fuel_type")
    body_type = st.selectbox("Body Type", filtered_body_types, key="body_type")
    variant_name = st.selectbox("Variant Name", unique_values_vn, key="variant_name")
    model_year = st.selectbox("Year of Manufacture", unique_values_my_sorted, key="model_year")
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    transmission = st.selectbox("Transmission", unique_values, key="transmission")
    owner_no = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)
    manufacturer = st.selectbox("Original Equipment Manufacturer (OEM)", unique_values_oem, key="manufacturer")
    car_model = st.selectbox("Car Model", unique_values_ml, key="car_model")
    insurance = st.selectbox("Insurance Validity", unique_values_iv, key="insurance")
    seats = st.selectbox("Seats", unique_values_seats, key="seats")

    if st.button("Estimate Price"):
        input_data = pd.DataFrame({
            'Transmission': [transmission],
            'Body Type': [body_type],
            'Fuel Type': [fuel_type],
            'OEM': [manufacturer],
            'Model': [car_model],
            'Variant Name': [variant_name],
            'City': [city],
            'Insurance Validity': [insurance],
            'Kilometers Driven': [kilometers_driven],
            'Model Year': [model_year],
            'Previous Owners': [owner_no],
            'Seats': [seats]
        })

        # One-hot encode input data
        input_data = pd.get_dummies(input_data, columns=categorical_columns)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Apply Min-Max Scaling
        input_data[['Kilometers Driven']] = scaler.transform(input_data[['Kilometers Driven']])

        # Make prediction
        predicted_price = model.predict(input_data)
        st.write(f"Estimated Price: ₹{predicted_price[0]:,.2f}")
        
        
#  ========================================= Predicts Price (not cached) ========================================

# import ast
# import pandas as pd
# import streamlit as st
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import joblib

# st.title("Car Dheko")
# st.image("./maruti-suzuki-brezza-brand.jpg")

# def extract_feature(data, key, default=None):
#     if isinstance(data, dict):
#         return data.get(key, default)
#     if isinstance(data, list):
#         for item in data:
#             if isinstance(item, dict) and item.get("key") == key:
#                 return item.get("value", default)
#     return default


# def parse_json(data):
#     if isinstance(data, dict):
#         return data
#     try:
#         return ast.literal_eval(data)
#     except (ValueError, SyntaxError):
#         return {}

# downloads_folder = os.path.join(os.getcwd(), "Data")  
# xlsx_files = [
#     "kolkata_cars.xlsx",
#     "jaipur_cars.xlsx",
#     "hyderabad_cars.xlsx",
#     "delhi_cars.xlsx",
#     "chennai_cars.xlsx",
#     "bangalore_cars.xlsx",
# ]

# all_features_df = pd.DataFrame()

# for file in xlsx_files:
#     file_path = os.path.join(downloads_folder, file)
#     df = pd.read_excel(file_path)

#     if "new_car_detail" in df.columns:
#         df["new_car_detail"] = df["new_car_detail"].apply(parse_json)

#         df_feature = pd.DataFrame({
#             "Price": df["new_car_detail"].apply(lambda x: extract_feature(x, "price")),
#             "Transmission": df["new_car_detail"].apply(lambda x: extract_feature(x, "transmission")),
#             "Body Type": df["new_car_detail"].apply(lambda x: extract_feature(x, "bt")),
#             "Fuel Type": df["new_car_detail"].apply(lambda x: extract_feature(x, "ft")),
#             "Kilometers Driven": df["new_car_detail"].apply(lambda x: (extract_feature(x, "km").replace(",", "") if extract_feature(x, "km") else "0")),
#             "Model Year": df["new_car_detail"].apply(lambda x: (str(extract_feature(x, "modelYear")).replace(",", "").strip() if extract_feature(x, "modelYear") is not None else "0")),
#             "Previous Owners": df["new_car_detail"].apply(lambda x: extract_feature(x, "ownerNo")),
#             "OEM": df["new_car_detail"].apply(lambda x: extract_feature(x, "oem")),
#             "Variant Name": df["new_car_detail"].apply(lambda x: extract_feature(x, "variantName")),
#             "City": file.split("_")[0],
#         })

#         def clean_year(value):
#             if pd.isna(value):
#                 return 0
#             return str(int(value))

#         all_features_df = pd.concat([all_features_df, df_feature])
#         all_features_df["Model Year"] = all_features_df["Model Year"].apply(clean_year)

#         def convert_price(value):
#             if isinstance(value, str):
#                 value = value.strip()
#                 if '₹' in value:
#                     value = value.replace('₹', '').replace(',', '')
#                     if 'Lakh' in value:
#                         value = float(value.replace('Lakh', '').strip()) * 100000
#                     elif 'Crore' in value:
#                         value = float(value.replace('Crore', '').strip()) * 10000000
#                     else:
#                         value = float(value)
#                 return value
#             return value

#         all_features_df['Price'] = all_features_df['Price'].apply(convert_price)

#     else:
#         st.write(f"The file {file} does not contain the 'new_car_detail' column.")


# all_features_df.dropna(axis=0, inplace=True)
# all_features_df.fillna(0, inplace=True)

# unique_values_city = all_features_df["City"].unique()
# unique_values_city_capitalized = [city.capitalize() for city in unique_values_city]
# with st.sidebar:
#     st.header("Car Features")
#     city = st.selectbox("City", unique_values_city_capitalized, key="city")
#     fuel_type = st.selectbox("Fuel Type", all_features_df["Fuel Type"].unique(), key="fuel_type")
#     body_type = st.selectbox("Body Type", all_features_df["Body Type"].unique(), key="body_type")
#     variant_name = st.selectbox("Variant Name", all_features_df["Variant Name"].unique(), key="variant_name")
#     model_year = st.selectbox("Year of Manufacture", all_features_df["Model Year"].unique(), key="model_year")
#     kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
#     transmission = st.selectbox("Transmission", all_features_df["Transmission"].unique(), key="transmission")
#     owner_no = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)
#     manufacturer = st.selectbox("OEM", all_features_df["OEM"].unique(), key="manufacturer")


# st.write(" Selected Car Details :")
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.write(f"**City**: {city}")
#     st.write(f"**Fuel Type**: {fuel_type}")
#     st.write(f"**Body Type**: {body_type}")
# with col2:
#     st.write(f"**Variant Name**: {variant_name}")
#     st.write(f"**Kilometers Driven**: {kilometers_driven}")
#     st.write(f"**Transmission**: {transmission}")
# with col3:
#     st.write(f"**Previous Owners**: {owner_no}")
#     st.write(f"**Manufacturer**: {manufacturer}")
#     st.write(f"**Year of Manufacture**: {model_year}")
# # st.button("Predict")
# categorical_columns = ['Transmission', 'Body Type', 'Fuel Type', 'OEM', 'Variant Name', 'City']
# all_features_df = pd.get_dummies(all_features_df, columns=categorical_columns, dtype='int')


# X = all_features_df.drop('Price', axis=1)
# y = all_features_df['Price']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     return y_pred

# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Save the trained model for future use
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')



# if st.button("Predict"):
#     input_data = {
#         "City": city,
#         "Fuel Type": fuel_type,
#         "Body Type": body_type,
#         "Model Year": model_year,
#         "Kilometers Driven": kilometers_driven,
#         "Transmission": transmission,
#         "Previous Owners": owner_no,
#         "OEM": manufacturer,
#         "Variant Name": variant_name,
#     }

#     input_df = pd.DataFrame([input_data])
#     input_df = pd.get_dummies(input_df, columns=categorical_columns, dtype='int')
#     input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
#     rf_prediction = rf_model.predict(input_df)
#     st.write(f" Predicted price: ₹{rf_prediction[0]:,.2f}")

# evaluate_model(rf_model, X_test, y_test)
