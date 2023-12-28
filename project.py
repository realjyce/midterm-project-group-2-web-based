import streamlit as st
#Streamlit CONFIG
config_file_path = "./.streamlit/config.toml"
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="chart_with_upwards_trend", page_title="Flood")
#Import CSS
with open('./css/style.css') as f:
    css = f.read()
    
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#Import Libs
import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
from sklearn.impute import SimpleImputer
from permetrics.regression import RegressionMetric #Metrics
from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor  # Import XGBoost
from streamlit_option_menu import option_menu #Extras
import streamlit_extras #Extras
import time #Extras
#Graphics & Data Visualisation
import seaborn as sns
from shapely.geometry import shape
import matplotlib.pyplot as plt
import plotly.express as px
#Heatmap
import leafmap.foliumap as leafmap
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
X = None
y = None
train_ratio = 0.8 
# MENU-ING AND TITLE
st.markdown(
    """
    <style>
        .streamlit-option-menu .stSelectbox .stText {
            font-size: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
menu = option_menu(
    menu_title="Web-Based Machine Learning Application",
    options=["Home", "Raw Data", "Model", "Predictions"],
    icons=["house", "database", "file-earmark-bar-graph","play"],
    menu_icon="robot",
    orientation="horizontal",
    )

# HOME
uploaded_file = None
if menu == "Home":
    def show_data(uploaded_file):
        st.title("")
        st.markdown('<h1 style="color: #170e45; width: 2000px; font-size: 205%; margin-bottom:20px; margin-top: -40px; "><span style="font-size:1.5em; display:inline-block; line-height:5px;margin-top:6px;"></span><span style="color:#006fff;">👾</span>  Hello User, upload Data in the <span style="color:#006fff">Raw Data</span> section to begin</h1>', unsafe_allow_html=True)
        if "data" not in st.session_state:
            st.warning("⚠️ Please upload a CSV file before proceeding. ⚠️")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("Data loaded successfully!")
            st.session_state.data = df

    def landing():
        st.title("🕮 Data Overview")
        df = st.session_state.data
        st.header("Head Overview Data")
        st.write(df.head())

    def datareq():
        if "data" not in st.session_state:
            show_data(uploaded_file)
        else:
            landing()

    if __name__ == "__main__":
        datareq()

if menu == "Raw Data":
    title = st.title('Input Data')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded file into a DataFrame
        df_upload = pd.read_csv(uploaded_file)

        # Use the uploaded file directly for subsequent operations
        st.session_state.data = df_upload
        st.session_state.uploaded_file = uploaded_file
        # User: selectbox the training and testing data ratio
        st.subheader('Select Training and Testing Data Ratio')
        ratio_option = st.selectbox("Data Ratio:", ["90:10", "80:20", "70:30", "60:40"])

        # Train-test split ratio
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
        X_train, X_test, y_train, y_test = train_test_split(df_upload.iloc[:, :-1], df_upload.iloc[:, -1], test_size=1 - train_ratio, random_state=42)
        st.subheader("Data Shape:")
        col1, col2, col3 = st.columns((1, 1, 3))

        # Training data shape + Display
        with col1:
            st.write("Training Data Shape:", X_train.shape)

        # Testing data shape + Display
        with col2:
            st.write("Testing Data Shape:", X_test.shape)

        # Display the training and testing data separately
        st.subheader("Training Data")
        tab1a, tab1b, tab1c = st.tabs(['Chart📈', 'DataFrame📄', 'Export📁'])
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
        tab2a, tab2b, tab2c = st.tabs(['Chart📈', 'DataFrame📄', 'Export📁'])
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
        # Rest of your code...

        if st.button('Rerun'):
            st.experimental_rerun()
            st.toast('Done!')

        st.toast('Done!')

# MODEL
if menu == "Model":
    uploaded_file = st.session_state.get('uploaded_file', None)
    st.header("Choose Machine Learning Model")
    model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
    if model_option == "Random Forest":
        model = RandomForestRegressor()
    elif model_option == "XGBoost":
        model = XGBRegressor(  
            learning_rate=0.1,      
            n_estimators=100,       
            max_depth=3,            
            min_child_weight=1,     
            subsample=0.8,          
            colsample_bytree=0.8,   
            objective='reg:squarederror',
            random_state=42
    )

    if st.button("Run Model"):
        if uploaded_file is not None:
            st.header("Run Model and Evaluate Results")
            df_upload = st.session_state.data
            st.toast('Running Code...')
            with st.spinner(text='Loading...'):
                time.sleep(1)
            X_train, X_test, y_train, y_test = train_test_split(df_upload.iloc[:, :-1], df_upload.iloc[:, -1], test_size=1 - train_ratio, random_state=42)

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
                print("Length of feature_names:", len(feature_names))
                print("Length of feature_importance:", len(feature_importance))
                # Update to use column names from the loaded CSV file
                feature_names = df_upload.columns
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

                # Bar chart
                importance_chart = sns.barplot(x="Importance", y="Feature", data=importance_df)
                st.pyplot(importance_chart.figure)

            st.toast('Done!')

            # Export predicted values in CSV format
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
    st.header("Predict on Study Area")
    st.subheader("Call New Data")
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Head DataFrame:")
        st.write(new_data.head())    
    
        # Model Selct
    st.subheader("Select Model for Predictions")
    prediction_model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    prediction_model = None
    if prediction_model_option == "Random Forest":
        prediction_model = RandomForestRegressor()
    elif prediction_model_option == "XGBoost":
        prediction_model = model = XGBRegressor(  
        learning_rate=0.1,      
        n_estimators=100,       
        max_depth=3,            
        min_child_weight=1,     
        subsample=0.8,          
        colsample_bytree=0.8,   
        objective='reg:squarederror',
          random_state=42)
    def load_model():
        model = None
        return model

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
                fig = px.imshow([new_data_predictions], title='Heatmap of Predicted Flood')
                st.plotly_chart(fig)
                # Download button for predictions (.csv)
                predictions_csv_data = convert_df(pd.DataFrame({"Predicted Flood": uploaded_data_predictions}))
            except NotFittedError:
                st.warning("The model has not been trained. Please click 'Train Model for Predictions'.")
        else:
            st.warning("Please train the model before making predictions.")
    

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
                new_data_predictions = prediction_model.predict(new_data)# If not, preprocess

                # Display predictions
                st.subheader("Predictions on Existing Data")
                df_results = pd.DataFrame({"Predicted Flood": new_data_predictions})
                st.write(df_results)
                # Download button for predictions (.csv)
                predictions_csv_data = convert_df(pd.DataFrame({"Predicted Flood": new_data_predictions}))
                
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=predictions_csv_data,
                    file_name='new_data_predictions.csv',
                    mime='text/csv',
                )
                fig = px.histogram(df_results, x='Predicted Flood', title='Density Map of Predicted Flood')
                st.plotly_chart(fig)
                fig = px.imshow([new_data_predictions], title='Heatmap of Predicted')
                st.plotly_chart(fig)
                    
            except NotFittedError:
                st.warning("The model has not been trained. Please click 'Train Model for Predictions'.")
            except Exception as e:
                st.stop()


            

    def load_model():
        model = None
        return model

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
                st.download_button(
                    label="💾 Download Density Map as ",
                    data=predictions_csv_data,
                    file_name='new_data_predictions.csv',
                    mime='text/csv',
                )
                fig = px.imshow([new_data_predictions], title='Heatmap of Predicted Flood')
                st.plotly_chart(fig)
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=predictions_csv_data,
                    file_name='new_data_predictions.csv',
                    mime='text/csv',
                )
                # Download button for predictions (.csv)
                predictions_csv_data = convert_df(pd.DataFrame({"Predicted Flood": uploaded_data_predictions}))
            except NotFittedError:
                st.warning("The model has not been trained. Please click 'Train Model for Predictions'.")

# Hide Watermark
hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)
