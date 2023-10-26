import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# MENU-ING AND TITLE
menu = st.sidebar.radio("Menu", ["Home", "Raw Data"])
if menu == "Home":
    title = st.title("Project 2B: Web-App Machine Learning with Python")
    st.balloons()
    st.write("'Hello World'")

# DATA CACHE FUNCTION
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

# Define y_train and y_test based on your dataset
y_train = None  
y_test = None  

# MENU: RAW DATA
if menu != "Raw Data":
    st.write('Shape of Dataset')
else:
    title = st.title('Raw Data for [Flood]')
    df = load_data('Data.csv')
    df_1 = df.iloc[:, :-1]

    # Allow user to select the training and testing data ratio
    ratio_option = st.selectbox("Select Training and Testing Data Ratio", ["90:10", "80:20", "70:30", "60:40"])

    # Map the selected ratio to a train-test split ratio
    if ratio_option == "90:10":
        train_ratio = 0.9
    elif ratio_option == "80:20":
        train_ratio = 0.8
    elif ratio_option == "70:30":
        train_ratio = 0.7
    elif ratio_option == "60:40":
        train_ratio = 0.6

    # Split the data into training and testing sets based on the selected ratio
    X = df_1  # Use df_1 as the data
    X_train, X_test, y_train, y_test = train_test_split(X, df.iloc[:, -1], test_size=1 - train_ratio, random_state=42)

    st.write("Data Shape:")

    # Display the shape of the training data
    st.write("Training Data Shape:", X_train.shape)

    # Display the shape of the testing data
    st.write("Testing Data Shape:", X_test.shape)

    # Display the training and testing data separately
    st.subheader("Training Data")
    st.dataframe(X_train)

    st.subheader("Testing Data")
    st.dataframe(X_test)

    st.button('Rerun')

    # Train the Random Forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    # Train the XGBoost model
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    # Make predictions with both models
    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)

    # Calculate performance metrics
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    mse_rf = mean_squared_error(y_test, rf_predictions)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, rf_predictions)

    mse_xgb = mean_squared_error(y_test, xgb_predictions)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, xgb_predictions)

    # Display model performance and predictions
    st.write("Random Forest Model Performance:")
    st.write(f"MSE: {mse_rf:.2f}")
    st.write(f"RMSE: {rmse_rf:.2f}")
    st.write(f"R2: {r2_rf:.2f}")

    st.write("XGBoost Model Performance:")
    st.write(f"MSE: {mse_xgb:.2f}")
    st.write(f"RMSE: {rmse_xgb:.2f}")
    st.write(f"R2: {r2_xgb:.2f}")

    st.write("Random Forest Predictions:")
    st.write(rf_predictions)

    st.write("XGBoost Predictions:")
    st.write(xgb_predictions)

    # Calculate RMSE, MAE, and R2 for Random Forest model
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_predictions))
    mae_rf = mean_absolute_error(y_test, rf_predictions)
    r2_rf = r2_score(y_test, rf_predictions)

    # Calculate RMSE, MAE, and R2 for XGBoost model
    rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    mae_xgb = mean_absolute_error(y_test, xgb_predictions)
    r2_xgb = r2_score(y_test, xgb_predictions)

    # Calculate metrics on training data as well
    rmse_rf_train = np.sqrt(mean_squared_error(y_train, rf_model.predict(X_train)))
    mae_rf_train = mean_absolute_error(y_train, rf_model.predict(X_train))
    r2_rf_train = r2_score(y_train, rf_model.predict(X_train))

    rmse_xgb_train = np.sqrt(mean_squared_error(y_train, xgb_model.predict(X_train)))
    mae_xgb_train = mean_absolute_error(y_train, xgb_model.predict(X_train))
    r2_xgb_train = r2_score(y_train, xgb_model.predict(X_train))

    # Display the metrics for both models on testing data
    st.write("Random Forest Model Performance on Testing Data:")
    st.write(f"RMSE: {rmse_rf:.2f}")
    st.write(f"MAE: {mae_rf:.2f}")
    st.write(f"R2: {r2_rf:.2f}")

    st.write("XGBoost Model Performance on Testing Data:")
    st.write(f"RMSE: {rmse_xgb:.2f}")
    st.write(f"MAE: {mae_xgb:.2f}")
    st.write(f"R2: {r2_xgb:.2f}")

    # Display the metrics for both models on training data
    st.write("Random Forest Model Performance on Training Data:")
    st.write(f"RMSE: {rmse_rf_train:.2f}")
    st.write(f"MAE: {mae_rf_train:.2f}")
    st.write(f"R2: {r2_rf_train:.2f}")

    st.write("XGBoost Model Performance on Training Data:")
    st.write(f"RMSE: {rmse_xgb_train:.2f}")
    st.write(f"MAE: {mae_xgb_train:.2f}")
    st.write(f"R2: {r2_xgb_train:.2f}")


hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)
