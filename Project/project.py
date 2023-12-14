import streamlit as st

#Streamlit CONFIG
config_file_path = "./project/config.toml"

#Import CSS
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="chart_with_upwards_trend", page_title="Flood")
with open('./css/style.css') as f:
    css = f.read()
    print(css)
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#Import Libs
import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
from permetrics.regression import RegressionMetric #Metrics
from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor  # Import XGBoost
from streamlit_option_menu import option_menu #Extras
import streamlit_extras #Extras
import time #Extras
#Graphics & Data Visualisation
import seaborn as sns 
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import plotly.express as px

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
df_new = load_data('./project/NewData.csv')

df_1 = df.iloc[:, :-1]
X = df_1
y = df.iloc[:, -1]  # Assuming last col is dependent var
#Download Display For Download: SUCCESS!
display_success = False

# Initialize variables outside (GLOBAL)
X_train, X_test, y_train, y_test = None, None, None, None
model = None
train_ratio = 0.8  # DEFAULT VALUE

# MENU-ING AND TITLE
menu = option_menu(
    menu_title=None,
    options=["Home", "Raw Data", "Model", "Predictions"],
    icons=["house", "database", "file-earmark-bar-graph","play"],
    menu_icon="cast",
    orientation="horizontal",
    )

if menu == "Home":
    st.title("Project 2B: Web-App Machine Learning with Python")
    st.header("Head Overview Data")
    st.write(df.head())
    st.header("Data Summary")
    st.write(df.describe())
    st.header("Data Values")
    st.write(df.value_counts(subset=None, normalize=False, sort=True, ascending=False, dropna=True))
    st.header("Data Keys")
    st.write(df.keys())

# RAW DATA
if menu == "Raw Data":
    title = st.title('[Flood]')
    st.dataframe(df)

    # User: selectbox the training and testing data ratio
    st.subheader('Select Training and Testing Data Ratio')
    ratio_option = st.selectbox("Data Ratio:", ["90:10", "80:20", "70:30", "60:40"])

    # Map the according ratio to a train-test split ratio
    if ratio_option == "90:10":
        train_ratio = 0.9
        st.toast('Running...')
    elif ratio_option == "80:20":
        train_ratio = 0.8
        st.toast('Running...')
    elif ratio_option == "70:30":
        train_ratio = 0.7
        st.toast('Running...')
    elif ratio_option == "60:40":
        train_ratio = 0.6
        st.toast('Running...')

    # Split the data into training and testing sets based on the selected ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
    st.subheader("Data Shape:")
    col1, col2, col3 = st.columns((1,1,3))

    # Training data shape + Display
    with col1:
        st.write("Training Data Shape:", X_train.shape)

     # Testing data shape + Display
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
        download1 = st.download_button(
            label="💾 Download Train.csv",
            data=train_data,
            file_name='train.csv',
            mime='text/csv',
        )
        if download1:
            st.success("Download Successful!")

    st.subheader("Testing Data")
    tab2a, tab2b, tab2c = st.tabs(['Chart📈','DataFrame📄','Export📁'])
    with tab2a:
        st.bar_chart(X_test)
    with tab2b:
        st.write(X_test)
    with tab2c:
        test_data = X_test.to_csv(index=False)
        download2 = st.download_button(
            label="💾Download Test.csv",
            data=test_data,
            file_name='test.csv',
            mime='text/csv',
        )
        if download2:
            st.success("Download Successful!")
        

    if st.button('Rerun'):
        st.experimental_rerun()
        st.toast('Done!')
        
    st.toast('Done!')

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
        # Split the data into training and testing sets based on the selected ratio, again.(for LOCAL scope)
        st.toast('Running Code...')
        with st.spinner(text='Loading...'):
            time.sleep(1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score

        # Metrics for training data
        rmse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Metrics for testing data
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
            error_hist_train = sns.histplot(y_train - y_train_pred, kde=True, figure=plt.figure(figsize=(7, 7)))
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
            
        st.toast('Done!')

        # Export predicted val in CSV format
        csv_data = convert_df(pd.DataFrame({"Actual (Test)": y_test, "Predicted (Test)": y_test_pred}))
        download3 = st.download_button(
            label="Download Predictions as CSV",
            data=csv_data,
            file_name='predictions.csv',
            mime='text/csv',
        ) 
        if download3: st.success("Download Successful!")
        
        
        
if menu == "Predictions":
   # Upload Section | NEWDATA
    st.header("Predict on New Data")
    st.header("Predictions")
    st.subheader("Head Overview NewData")
    st.write(df_new.head())

    # Model selection for prediction
    st.subheader("Select Model for Predictions")
    prediction_model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    prediction_model = None
    if prediction_model_option == "Random Forest":
        prediction_model = RandomForestRegressor()
    elif prediction_model_option == "XGBoost":
        prediction_model = XGBRegressor()

    if prediction_model is not None:
        # Train the model, before predicting
        if st.button("Train Model for Predictions"):
            # If cols in df_new are == to the training data
            with st.spinner(text='Loading...'):
                time.sleep(1)
            st.toast("Running...")
            X_train, _, y_train, _ = train_test_split(X, y, test_size=1 - train_ratio, random_state=42) # If not, repeat training process
            prediction_model.fit(X_train, y_train)
            st.toast("Done!")

        # Check model fitting before predictions
        if hasattr(prediction_model, 'predict'): #has attribute of 'predict'
            try:
                # If cols in df_new are == to the training data
                new_data_predictions = prediction_model.predict(df_new)# If not, preprocess

                # Display predictions
                st.subheader("Predictions on New Data")
                df_results = pd.DataFrame({"Predicted Flood": new_data_predictions})
                st.write(df_results)
                # DENSITY MAP with plotly
                fig = px.histogram(df_results, x='Predicted Flood', title='Density Map of Predicted Flood')
                st.plotly_chart(fig)
                # Download button for predictions (.csv)
                predictions_csv_data = convert_df(pd.DataFrame({"Predicted Flood": new_data_predictions}))
                st.download_button(
                    label="Download Predictions on NewData as CSV",
                    data=predictions_csv_data,
                    file_name='new_data_predictions.csv',
                    mime='text/csv',
                )

            except NotFittedError:
                st.warning("The model has not been trained. Please click 'Train Model for Predictions'.")

        else:
            st.warning("Please train the model before making predictions.")

    def load_model():
        model = None
        return model

    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        if model is None:
            st.success("Uploaded")
            # Train the model, before predicting
        if st.button("Train Uploaded Model for Predictions"):
            # If cols in df_new are == to the training data
            with st.spinner(text='Loading...'):
                time.sleep(1)
            st.toast("Running...")
            X_train, _, y_train, _ = train_test_split(X, y, test_size=1 - train_ratio, random_state=42) # If not, repeat training process
            prediction_model.fit(X_train, y_train)
            st.toast("Done!")

        # Check model fitting before predictions
        if hasattr(prediction_model, 'predict'): #has attribute of 'predict'
            try:
                # If cols in df_new are == to the training data
                uploaded_data_predictions = prediction_model.predict(new_data)# If not, preprocess

                # Display predictions
                st.subheader("Predictions on uploaded New Data")
                df_results = pd.DataFrame({"Predicted Flood": uploaded_data_predictions})
                st.write(df_results)
                fig = px.histogram(df_results, x='Predicted Flood', title='Density Map of Predicted Flood')
                st.plotly_chart(fig)
                # Download button for predictions (.csv)
                predictions_csv_data = convert_df(pd.DataFrame({"Predicted Flood": uploaded_data_predictions}))
            except NotFittedError:
                st.warning("The model has not been trained. Please click 'Train Model for Predictions'.")

        
st.markdown(
    """
    <div style="font-size: 13px;position: absolute; left:44%; bottom: -180px; width: 13%; text-align: center; color: #FFFFFF; box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.259); border-radius: 15px; background: #006fff;">
        <u>Made by Group 2B🩵</u>
    </div>
    """,
    unsafe_allow_html=True,
)
# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)