import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="chart_with_upwards_trend", page_title="Flood")

with open('./project/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

import altair as alt
import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
import seaborn as sns
from permetrics.regression import RegressionMetric
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # Import XGBoost
from streamlit_option_menu import option_menu
import streamlit_extras

# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Function to convert DataFrame to CSV data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# DATA CACHING
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('./project/Data.csv')

df_1 = df.iloc[:, :-1]
X = df_1
y = df.iloc[:, -1]  # Assuming the last column is the dependent variable

# Initialize variables outside if statements
X_train, X_test, y_train, y_test = None, None, None, None
model = None
train_ratio = 0.8  # Default value, can be adjusted based on preference

# MENU-ING AND TITLE
menu = option_menu(
    menu_title=None,
    options=["Home", "Raw Data", "Model"],
    icons=["house", "database", "play"],
    menu_icon="cast",
    orientation="horizontal",
    )

if menu == "Home":
    st.title("Project 2B: Web-App Machine Learning with Python")
    st.header("Head Overview Data")
    st.write(df.head())
    st.header("Data Summary")
    st.write(df.describe())

# RAW DATA
if menu == "Raw Data":
    title = st.title('[Flood]')
    st.dataframe(df)

    # Allow user to select the training and testing data ratio
    st.subheader('Select Training and Testing Data Ratio')
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
    st.subheader("Data Shape:")
    col1, col2, col3 = st.columns((1,1,3))

    # Display the shape of the training data
    with col1:
        st.write("Training Data Shape:", X_train.shape)

    # Display the shape of the testing data
    with col2:
        st.write("Testing Data Shape:", X_test.shape)

    # Display the training and testing data separately
    st.subheader("Training Data")
    tab1a, tab1b, tab1c = st.tabs(['Chart📈','DataFrame📄','Export📁'])
    with tab1a:
        st.bar_chart(X_train)
    with tab1b:
        st.write(X_train)
    with tab1c:
        train_data = X_train.to_csv(index=False)
        st.download_button(
            label="💾 Download Train.csv",
            data=train_data,
            file_name='train.csv',
            mime='text/csv',
        )

    st.subheader("Testing Data")
    tab2a, tab2b, tab2c = st.tabs(['Chart📈','DataFrame📄','Export📁'])
    with tab2a:
        st.bar_chart(X_test)
    with tab2b:
        st.write(X_test)
    with tab2c:
        test_data = X_test.to_csv(index=False)
        st.download_button(
            label="💾Download Test.csv",
            data=test_data,
            file_name='test.csv',
            mime='text/csv',
        )

    if st.button('Rerun'):
        st.experimental_rerun()

# MODEL
if menu == "Model":
    st.header("Choose Machine Learning Model")
    model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    if model_option == "Random Forest":
        model = RandomForestRegressor()

    if model_option == "XGBoost":
        model = XGBRegressor()  # Initialize XGBoost model

    st.header("Run Model and Evaluate Results")
    if st.button("Run Model"):
        # Split the data into training and testing sets based on the selected ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score

        # Evaluation metrics for training data
        rmse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Evaluation metrics for testing data
        rmse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        st.divider()
        
        st.header("Evaluate Training Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f'{rmse_train:.5f}')
        col2.metric("MAE", f'{mae_train:.5f}')
        col3.metric("R2 Score", f'{r2_train:.5f}')
        
        st.divider()
        
        st.header("Evaluate Testing Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f'{rmse_test:.5f}')
        col2.metric("MAE", f'{mae_test:.5f}')
        col3.metric("R2 Score", f'{r2_test:.5f}')

        st.divider()

        col1, col2, col3 = st.columns((1, 1, 1))

        # Histogram of errors for training data
        with col1:
            st.subheader("Histogram of Errors (Training)")
            error_hist_train = sns.histplot(y_train - y_train_pred, kde=True)
            st.pyplot(error_hist_train.figure)

        # Histogram of errors for testing data
        with col2:
            st.subheader("Histogram of Errors (Testing)")
            error_hist_test = sns.histplot(y_test - y_test_pred, kde=True)
            st.pyplot(error_hist_test.figure)

        # Feature Importance
        with col3:
            st.subheader("Feature Importance")

            # Update to use column names from the loaded CSV file
            feature_names = df_1.columns
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

            # Bar chart
            importance_chart = sns.barplot(x="Importance", y="Feature", data=importance_df)
            st.pyplot(importance_chart.figure)

        # Export predicted values to CSV
        csv_data = convert_df(pd.DataFrame({"Actual (Test)": y_test, "Predicted (Test)": y_test_pred}))
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_data,
            file_name='predictions.csv',
            mime='text/csv',
        )

# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)