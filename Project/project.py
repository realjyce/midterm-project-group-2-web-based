import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TITLE
st.title("Project 2B: Web-App Machine Learning with Python")

# TEST
readData = pd.read_csv('Data.csv')
st.write(readData)
