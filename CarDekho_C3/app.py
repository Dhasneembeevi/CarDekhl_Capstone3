import ast
import pandas as pd
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


@st.cache_data
def load_and_train_model():
    downloads_folder = os.path.join(os.getcwd(), "Data")  
    xlsx_files = [
        "kolkata_cars.xlsx",
        "jaipur_cars.xlsx",
        "hyderabad_cars.xlsx",
        "delhi_cars.xlsx",
        "chennai_cars.xlsx",
        "bangalore_cars.xlsx",
    ]

    all_features_df = pd.DataFrame()

    def extract_feature(data, key, default=None):
        if isinstance(data, dict):
            return data.get(key, default)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("key") == key:
                    return item.get("value", default)
        return default

    def parse_json(data):
        if isinstance(data, dict):
            return data
        try:
            return ast.literal_eval(data)
        except (ValueError, SyntaxError):
            return {}

    for file in xlsx_files:
        file_path = os.path.join(downloads_folder, file)
        df = pd.read_excel(file_path)

        if "new_car_detail" in df.columns:
            df["new_car_detail"] = df["new_car_detail"].apply(parse_json)

            df_feature = pd.DataFrame({
                "Price": df["new_car_detail"].apply(lambda x: extract_feature(x, "price")),
                "Transmission": df["new_car_detail"].apply(lambda x: extract_feature(x, "transmission")),
                "Body Type": df["new_car_detail"].apply(lambda x: extract_feature(x, "bt")),
                "Fuel Type": df["new_car_detail"].apply(lambda x: extract_feature(x, "ft")),
                "Kilometers Driven": df["new_car_detail"].apply(lambda x: (extract_feature(x, "km").replace(",", "") if extract_feature(x, "km") else "0")),
                "Model Year": df["new_car_detail"].apply(lambda x: (str(extract_feature(x, "modelYear")).replace(",", "").strip() if extract_feature(x, "modelYear") is not None else "0")),
                "Previous Owners": df["new_car_detail"].apply(lambda x: extract_feature(x, "ownerNo")),
                "OEM": df["new_car_detail"].apply(lambda x: extract_feature(x, "oem")),
                "Variant Name": df["new_car_detail"].apply(lambda x: extract_feature(x, "variantName")),
                "City": file.split("_")[0],
            })

            def clean_year(value):
                if pd.isna(value):
                    return 0
                return str(int(value))

            all_features_df = pd.concat([all_features_df, df_feature])
            all_features_df["Model Year"] = all_features_df["Model Year"].apply(clean_year)

            def convert_price(value):
                if isinstance(value, str):
                    value = value.strip()
                    if '₹' in value:
                        value = value.replace('₹', '').replace(',', '')
                        if 'Lakh' in value:
                            value = float(value.replace('Lakh', '').strip()) * 100000
                        elif 'Crore' in value:
                            value = float(value.replace('Crore', '').strip()) * 10000000
                        else:
                            value = float(value)
                    return value
                return value

            all_features_df['Price'] = all_features_df['Price'].apply(convert_price)

    all_features_df.dropna(axis=0, inplace=True)
    
    all_features_df.fillna(0, inplace=True)

   
    categorical_columns = ['Transmission', 'Body Type', 'Fuel Type', 'OEM', 'Variant Name', 'City']
    all_features_df_dum = pd.get_dummies(all_features_df, columns=categorical_columns, dtype='int')

    X = all_features_df_dum.drop('Price', axis=1)
    y = all_features_df_dum['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

   
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
   
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = rf_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)


    return rf_model, X, all_features_df, mse, mae,r2


rf_model, X, all_features_df, mse, mae,r2 = load_and_train_model()


st.title("Car Dheko")
st.image("./maruti-suzuki-brezza-brand.jpg")
all_features_df.dropna(axis=0, inplace=True)

all_features_df = all_features_df[all_features_df["Body Type"] != '']

unique_values_city = all_features_df["City"].unique()
unique_values_city_capitalized = [city.capitalize() for city in unique_values_city]
with st.sidebar:
    st.header("Car Features")
    city = st.selectbox("City", unique_values_city_capitalized, key="city")
    fuel_type = st.selectbox("Fuel Type", all_features_df["Fuel Type"].unique(), key="fuel_type")
    body_type = st.selectbox("Body Type", all_features_df["Body Type"].unique(), key="body_type")
    variant_name = st.selectbox("Variant Name", all_features_df["Variant Name"].unique(), key="variant_name")
    model_year = st.selectbox("Year of Manufacture", all_features_df["Model Year"].unique(), key="model_year")
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    transmission = st.selectbox("Transmission", all_features_df["Transmission"].unique(), key="transmission")
    owner_no = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)
    manufacturer = st.selectbox("OEM", all_features_df["OEM"].unique(), key="manufacturer")


st.write("Selected Car Details:")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**City**: {city}")
    st.write(f"**Fuel Type**: {fuel_type}")
    st.write(f"**Body Type**: {body_type}")

with col2:
    st.write(f"**Variant Name**: {variant_name}")
    st.write(f"**Kilometers Driven**: {kilometers_driven}")
    st.write(f"**Transmission**: {transmission}")
with col3:
    st.write(f"**Previous Owners**: {owner_no}")
    st.write(f"**Manufacturer**: {manufacturer}")
    st.write(f"**Year of Manufacture**: {model_year}")

st.write(f" MAE: ₹{mae:.2f}")
st.write(f" MSE: ₹{mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
if st.button("Predict"):
    
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data = {
        "City": city,
        "Fuel Type": fuel_type,
        "Body Type": body_type,
        "Model Year": model_year,
        "Kilometers Driven": kilometers_driven,
        "Transmission": transmission,
        "Previous Owners": owner_no,
        "OEM": manufacturer,
        "Variant Name": variant_name,
    }

    input_df = pd.DataFrame([input_data])
    input_df_dummies = pd.get_dummies(input_df)
    input_df_dummies = input_df_dummies.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df_dummies)
    rf_prediction = rf_model.predict(input_scaled)
    st.write(f"Predicted price: ₹{rf_prediction[0]:,.2f}")

