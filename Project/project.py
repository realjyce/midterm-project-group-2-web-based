import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from permetrics.regression import RegressionMetric  

# MENU-ING AND TITE
menu = st.sidebar.radio("Menu",["Home", "Raw Data"])
if menu=="Home":
    title=st.title("Project 2B: Web-App Machine Learning with Python")
    st.balloons()
    st.write("'Hello World'")

# DATA CACHE FUNCTION
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

# MENU:HOME

# TEST
diamond = sns.load_dataset('diamonds')

# MENU: RAW DATA
if menu!="Raw Data":
        st.write('Shape of Dataset', diamond.shape)
else:
    title = st.title('Raw Data for [Flood]')
    df = load_data('./project/Data.csv')
    st.dataframe(df)
    st.button('Rerun')
    
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)