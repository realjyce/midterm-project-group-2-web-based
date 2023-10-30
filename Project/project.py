import streamlit as st
st.set_page_config(layout="wide",initial_sidebar_state = "expanded")
from xgboost import XGBRegressor
import pandas as pd
import ssl
import seaborn as sns
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Import CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

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
    
#MODEL
if menu=="Model":
    # Creating Scatter Plot for Model
    y_predict = model.predict(X_test)
    sns.scatterplot(x=X_test, y=y_predict)
    plt.xlabel('Raw Y')
    plt.ylabel('Predicted Y')
    st.pyplot()


# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}


        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)