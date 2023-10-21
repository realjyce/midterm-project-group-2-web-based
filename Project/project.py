import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from permetrics.regression import RegressionMetric  

# TITLE
st.title("Project 2B: Web-App Machine Learning with Python")

# TEST
readData = pd.read_csv('./project/Data.csv')
diamond = sns.load_dataset('diamonds')
st.write('Shape of Dataset', diamond.shape)
menu = st.sidebar.radio("Menu",["Home", "Raw Data"])
if menu=="Raw Data":
        st.write(readData)

