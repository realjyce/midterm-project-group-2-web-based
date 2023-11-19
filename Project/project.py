import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

with open('./style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
import seaborn as sns
from permetrics.regression import RegressionMetric
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # Import XGBoost
from streamlit_option_menu import option_menu

# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# DATA CACHING
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('./Data.csv')

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
    options=["Home", "Raw Data", "Model", "New Data"],
    icons=["house", "database", "play", "new_data"],
    menu_icon="cast",
    orientation="horizontal",
    )

if menu == "Home":
    st.title("Project 2B: Web-App Machine Learning with Python")
    st.header("Head Overview Data")
    st.write(df.head())

# RAW DATA
if menu == "Raw Data":
    title = st.title('Raw Data for [Flood]')
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
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score

        # Evaluation metrics
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        col1.metric("RMSE", value=rmse)
        col2.metric("MAE", value=mae)
        col3.metric("R2 Score", value=r2)
        st.divider()
        
        col1, col2 = st.columns((1,1))

        # Histogram of errors
        with col1:
            st.subheader("Histogram of Errors")
            error_hist = sns.histplot(y_test - y_pred, kde=True)
            st.pyplot(error_hist.figure)

        # Feature Importance
        with col2:
            st.subheader("Feature Importance")
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": df_1.columns, "Importance": feature_importance})
            importance_chart = sns.barplot(x="Importance", y="Feature", data=importance_df)
            st.pyplot(importance_chart.figure)
#New Data Upload
if menu == "New Data":
   # New Data Upload Section
    st.header("Predict on New Data")
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
    # Perform any necessary preprocessing to align with the model's requirements
    # Assuming the preprocessing steps are done to align the columns and data type
        if model is None:
            st.warning("Please select and run a model before making predictions.")
        else:
            predictions = model.predict(new_data.drop(['latitude', 'longitude'], axis=1))
            new_data['Predicted_Output'] = predictions

            # Displaying Map with Heatmap Layer
            st.header("Density Heat Map based on Predictions")
            st.map(new_data)

    if st.button("Predict on New Data"):
        predictions = model.predict(new_data.drop(['latitude', 'longitude'], axis=1))
        new_data['Predicted_Output'] = predictions

        # Displaying Map with Heatmap Layer
        st.header("Density Heat Map based on Predictions")
        st.map(new_data)


# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}


        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)