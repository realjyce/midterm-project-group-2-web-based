import streamlit as st
config_file_path = "./.streamlit/config.toml"
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="chart_with_upwards_trend", page_title="Flood")
with open('./css/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

class _SessionState:
    def __init__(self):
        self.prediction_model = None
        self.X_train = None
        self.y_train = None

session_state = _SessionState()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
from sklearn.impute import SimpleImputer
from permetrics.regression import RegressionMetric
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor
import streamlit_option_menu
import streamlit_extras
import time
import seaborn as sns
from shapely.geometry import shape
import matplotlib.pyplot as plt
import plotly.express as px
import leafmap.foliumap as leafmap
import seaborn as sns
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap

ssl._create_default_https_context = ssl._create_unverified_context

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

X = None
y = None
train_ratio = 0.8

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

menu = streamlit_option_menu.option_menu(
    menu_title="Web-Based Machine Learning Application",
    options=["Home", "Raw Data", "Model", "Predictions"],
    icons=["house", "database", "file-earmark-bar-graph", "play"],
    menu_icon="robot",
    orientation="horizontal",
)

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
        st.header("Head Data")
        st.write(df.head())
        st.header("Data Values")
        st.write(df.value_counts(subset=None, normalize=False, sort=True, ascending=False, dropna=True))
        st.header("Data Description & Key")
        col1, col2 = st.columns(2)
        col1.write(df.describe())
        col2.write(df.keys())

    def datareq():
        if "data" not in st.session_state:
            show_data(uploaded_file)
        else:
            landing()

    if __name__ == "__main__":
        datareq()

if menu == "Raw Data":
    title = st.title('📥Input Data')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        
        st.session_state.data = df_upload
        st.session_state.uploaded_file = uploaded_file

        st.subheader('Select Training and Testing Data Ratio')
        ratio_option = st.selectbox("Data Ratio:", ["90:10", "80:20", "70:30", "60:40"])

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

        st.subheader('Select Independent and Dependent Columns')
        independent_col = st.multiselect("Independent Columns:", df_upload.columns[:-1])
        st.write("Selected Independent Columns:", independent_col)
        if len(df_upload.columns) > 1:
            dependent_col_options = df_upload.columns.tolist()
            dependent_col = st.selectbox("Dependent Column:", dependent_col_options)
            st.write("Selected Dependent Column:", dependent_col)
        else:
            st.warning("Not enough columns to select a dependent variable.")
            st.stop()

        X = df_upload[independent_col]
        y = df_upload[dependent_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

        st.subheader("Data Shape:")
        col1, col2, col3 = st.columns((1, 1, 3))

        with col1:
            st.write("Training Data Shape:", X_train.shape)

        with col2:
            st.write("Testing Data Shape:", X_test.shape)

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

        if st.button('Rerun'):
            st.experimental_rerun()
            st.toast('Done!')

        st.toast('Done!')



if menu == "Model":
    uploaded_file = st.session_state.get('uploaded_file', None)
    st.title("⚙️Choose Machine Learning Model")
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
            with st.spinner(text='Evaluating...'):
                time.sleep(1)
            X_train, X_test, y_train, y_test = train_test_split(df_upload.iloc[:, :-1], df_upload.iloc[:, -1], test_size=1 - train_ratio, random_state=42)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            rmse_train = mean_squared_error(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

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

            with col1:
                st.subheader("Histogram of Errors (Training)")
                error_hist_train = sns.histplot(y_train - y_train_pred, kde=True, figure=plt.figure(figsize=(7, 7)))
                st.pyplot(error_hist_train.figure)

            with col2:
                st.subheader("Histogram of Errors (Testing)")
                error_hist_test = sns.histplot(y_test - y_test_pred, kde=True)
                st.pyplot(error_hist_test.figure)

            with col3:
                st.subheader("Feature Importance")

                feature_names = df_upload.columns
                feature_importance = model.feature_importances_

                print("Length of feature_names:", len(feature_names))
                print("Length of feature_importance:", len(feature_importance))

                if len(feature_importance) < len(feature_names):
                    feature_importance = np.concatenate([feature_importance, np.zeros(len(feature_names) - len(feature_importance))])

                importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
                print("Length of importance_df:", len(importance_df))

                importance_chart = sns.barplot(x="Importance", y="Feature", data=importance_df)
                st.pyplot(importance_chart.figure)

            st.toast('Done!')

            csv_data = convert_df(pd.DataFrame({"Actual (Test)": y_test, "Predicted (Test)": y_test_pred}))
            download3 = st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name='predictions.csv',
                mime='text/csv',
            )
            if download3:
                st.success("Download Successful!")

df_new = None
X_train_new = None

if menu == "Predictions":
    st.header("Predict on New Data")
    st.header("Predictions")

    uploaded_file_new = st.file_uploader("Upload CSV file for prediction on new data", type=["csv"])

    if uploaded_file_new is not None:
        df_new = pd.read_csv(uploaded_file_new)
        st.success("Successfully uploaded new data!")
        st.session_state.df_new = df_new

    st.subheader("Select Model for Predictions on New Data")
    prediction_model_option_new = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    prediction_model_new = None
    if prediction_model_option_new == "Random Forest":
        prediction_model_new = RandomForestRegressor()
    elif prediction_model_option_new == "XGBoost":
        prediction_model_new = XGBRegressor(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )

    if prediction_model_new is not None and df_new is not None:
        if st.button("Train Model for Predictions"):
            with st.spinner(text='Loading...'):
                time.sleep(1)
            st.toast("Predicting...")

            X_train_new, _, y_train_new, _ = train_test_split(df_new.iloc[:, :-1], df_new.iloc[:, -1], test_size=1 - train_ratio, random_state=42)
            
            prediction_model_new.fit(X_train_new, y_train_new)
            st.session_state.new_data_predictions = pd.DataFrame({"Predicted Flood": prediction_model_new.predict(X_train_new)})
            
            st.toast("Done!")

        if hasattr(prediction_model_new, 'predict'):
            try:
                if df_new is not None and X_train_new is not None:
                    new_data_predictions = prediction_model_new.predict(X_train_new)
                    col1, col2 = st.columns((1, 2))
                    with col1:
                        st.subheader("Predictions on New Data")
                        df_results = pd.DataFrame({"Predicted Flood": new_data_predictions})
                        st.write(df_results)
                        predictions_csv_data = convert_df(pd.DataFrame({"Prediction": new_data_predictions}))
                    st.download_button(
                        label="Download Predictions on NewData as .csv file",
                        data=predictions_csv_data,
                        file_name='new_data_predictions.csv',
                        mime='text/csv',
                    )
                    with col2:
                        fig = px.histogram(df_results, x='Prediction', title='Density Map of the Prediction')
                        st.plotly_chart(fig)

                else:
                    st.warning("Train the model before begin predicting 🤚.")
            except NotFittedError:
                st.warning("The model has not been trained yet. Press Train!")
        else:
            st.warning("Train the model predicting \(￣︶￣*\).")

        st.subheader("Heatmap of Prediction")

        col1, col2, col3 = st.columns((1, 1, 2))

        with col1:
            lat = st.selectbox("Select LAT:", df_new.columns)

        with col2:
            lon = st.selectbox("Select LON:", df_new.columns)

        with col3:
            if 'new_data_predictions' in st.session_state and isinstance(st.session_state.new_data_predictions, pd.DataFrame):
                pred_col = st.selectbox("Select Column:", st.session_state.new_data_predictions.columns, key="heatmap_column")
            else:
                st.warning("Prediction Required!")

        map = [df_new[lat].mean(), df_new[lon].mean()]
        heatmap = folium.Map(location=map, zoom_start=10)

        if 'new_data_predictions' in st.session_state and isinstance(st.session_state.new_data_predictions, pd.DataFrame):
            if lat in df_new.columns and lon in df_new.columns and pred_col in st.session_state.new_data_predictions.columns:
                data_heat = []
                norm = (st.session_state.new_data_predictions[pred_col] - st.session_state.new_data_predictions[pred_col].min()) / (st.session_state.new_data_predictions[pred_col].max() - st.session_state.new_data_predictions[pred_col].min())

                for i in df_new.index:
                    if lat in df_new.columns and lon in df_new.columns:
                        intensity = norm.get(i, None)
                        if intensity is not None:
                            data_heat.append([df_new.at[i, lat], df_new.at[i, lon], intensity])
                    else:
                        print(f"Missing columns in DataFrame: {df_new.columns}")

                if data_heat:
                    HeatMap(data_heat).add_to(heatmap)
                    folium_static(heatmap)
                else:
                    st.warning("Insufficient Data.")
            else:
                st.warning("Columns not found!")
        else:
            st.warning("Complete Prediction To Unlock The Heatmap 🗺️")



hide_made_with_streamlit = """
    <style>
        #MainMenu{visibility: hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_made_with_streamlit, unsafe_allow_html=True)
