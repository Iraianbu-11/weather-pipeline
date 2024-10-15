from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

# Define default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'weather_model_training',
    default_args=default_args,
    description='A DAG for training weather prediction models',
    schedule_interval=timedelta(days=1),
)

# Global variable for dataset
df = None

# Load Dataset
def download_dataset():
    global df
    file_path = '/opt/airflow/include/weather.csv'  # Relative path to the dataset
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise  # Raise the error to fail the task

# Preprocess Dataset
def preprocess_dataset():
    global df
    if df is None:
        raise ValueError("DataFrame 'df' is not initialized. Make sure the download_dataset task is successful.")

    df.drop(columns=['FullDate', 'Code', 'Location'], inplace=True)

    # Label Encoding
    le = LabelEncoder()
    df['City'] = le.fit_transform(df['City'])
    df['State'] = le.fit_transform(df['State'])
    df['WindDirection'] = le.fit_transform(df['WindDirection'])

    # Add new feature for temperature difference
    df['TempDiff'] = df['MaxTemp'] - df['MinTemp']

# Scale Dataset
def scale_dataset():
    global X_train_scaled, X_test_scaled, y_train, y_test
    if df is None:
        raise ValueError("DataFrame 'df' is not initialized. Make sure the previous tasks are successful.")

    X = df.drop('Precipitation', axis=1)  # Features
    y = df['Precipitation']  # Target

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Train Model
def weather_model():
    global X_train_scaled, y_train
    if X_train_scaled is None or y_train is None:
        raise ValueError("Training data not initialized. Make sure previous tasks are successful.")

    # Gradient Boosting
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr_model.fit(X_train_scaled, y_train)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

# Define tasks
t1 = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag,
)

t2 = PythonOperator(
    task_id='preprocess_dataset',
    python_callable=preprocess_dataset,
    dag=dag,
)

t3 = PythonOperator(
    task_id='scale_dataset',
    python_callable=scale_dataset,
    dag=dag,
)

t4 = PythonOperator(
    task_id='weather_model',
    python_callable=weather_model,
    dag=dag,
)

# Set task dependencies
t1 >> t2 >> t3 >> t4
