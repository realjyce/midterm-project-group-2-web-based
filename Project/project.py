import streamlit as st
#Streamlit Wide Mode
st.set_page_config(layout="wide",initial_sidebar_state = "expanded")

#Import CSS File
with open('./project/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#Libraries and Modules
from xgboost import XGBRegressor
import pandas as pd
import ssl
import seaborn as sns
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# DATA CACHING
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('./project/Data.csv')

df_1 = df.iloc[:, :-1]
X = df_1
y_train = None
y_test = None

# MENU-ING AND TITLE
menu = st.sidebar.radio("Menu",["Home", "Raw Data", "Model"])
if menu=="Home":
    st.title("Project 2B: Web-App Machine Learning with Python")
    st.header("Head Overview Data")
    st.write(df.head())
#RAW DATA
if menu == "Raw Data":
    title = st.title('Raw Data for [Flood]')
    st.dataframe(df)
    # Allow user to select the training and testing data ratio
    st.subheader('Select Training and Testing')
    ratio_option = st.selectbox("Data Ratio:", ["90:10", "80:20", "70:30", "60:40"])

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
    X_train, X_test, _, _ = train_test_split(X, X, test_size=1 - train_ratio, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, df.iloc[:, -1], test_size=1 - train_ratio, random_state=42)

    st.subheader("Data Shape:")
    
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

    X_train, X_test, y_train, y_test = train_test_split(X, df.iloc[:, -1], test_size=1 - train_ratio, random_state=42)
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
    st.header('Model Performance')
    
    st.write("Random Forest Model Performance:")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE:", f'{mse_rf:.2f}')
    col2.metric("RMSE:", f'{rmse_rf:.2f}')
    col3.metric("R2", f'{r2_rf:.2f}')

    st.write("XGBoost Model Performance:")
    col1a, col2a, col3a = st.columns(3)
    col1a.metric("MSE:", f'{mse_xgb:.2f}')
    col2a.metric("RMSE:", f'{rmse_xgb:.2f}')
    col3a.metric("R2", f'{r2_xgb:.2f}')

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
    st.button('Rerun All')
    
    # MODEL
if menu == "Model":
    st.sidebar.header("Choose Machine Learning Model")
    model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
    
    if model_option == "Random Forest":
        model = RandomForestRegressor()

    if model_option == "XGBoost":
        model = XGBRegressor()  # Initialize XGBoost model

    st.sidebar.header("Run Model and Evaluate Results")
    if st.sidebar.button("Run Model"):
        # Split the data into training and testing sets based on the selected ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score

        # Evaluation metrics
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"R^2 Score: {r2}")

        # Histogram of errors
        st.subheader("Histogram of Errors")
        error_hist = sns.histplot(y_test - y_pred, kde=True)
        st.pyplot(error_hist.figure)

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": df_1.columns, "Importance": feature_importance})
        importance_chart = sns.barplot(x="Importance", y="Feature", data=importance_df)
        st.pyplot(importance_chart.figure)
# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}


        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)