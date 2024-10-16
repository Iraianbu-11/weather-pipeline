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
from io import StringIO

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

# Load Dataset
def download_dataset(**kwargs):
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/Iraianbu-11/weather-pipeline/main/weather.csv")
        logging.info("Dataset loaded successfully.")
        # Save DataFrame to XCom
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        kwargs['ti'].xcom_push(key='weather_data', value=csv_buffer.getvalue())
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Preprocess Dataset
def preprocess_dataset(**kwargs):
    csv_data = kwargs['ti'].xcom_pull(key='weather_data')
    df = pd.read_csv(StringIO(csv_data))

    df.drop(columns=['FullDate', 'Code', 'Location'], inplace=True)

    # Label Encoding
    le = LabelEncoder()
    df['City'] = le.fit_transform(df['City'])
    df['State'] = le.fit_transform(df['State'])
    df['WindDirection'] = le.fit_transform(df['WindDirection'])

    # Add new feature for temperature difference
    df['TempDiff'] = df['MaxTemp'] - df['MinTemp']

    # Save preprocessed data to XCom
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    kwargs['ti'].xcom_push(key='preprocessed_data', value=csv_buffer.getvalue())

# Scale Dataset
def scale_dataset(**kwargs):
    csv_data = kwargs['ti'].xcom_pull(key='preprocessed_data')
    df = pd.read_csv(StringIO(csv_data))

    X = df.drop('Precipitation', axis=1)  # Features
    y = df['Precipitation']  # Target

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaled data to XCom
    kwargs['ti'].xcom_push(key='X_train_scaled', value=X_train_scaled.tolist())
    kwargs['ti'].xcom_push(key='X_test_scaled', value=X_test_scaled.tolist())
    kwargs['ti'].xcom_push(key='y_train', value=y_train.tolist())
    kwargs['ti'].xcom_push(key='y_test', value=y_test.tolist())

# Train Model
def weather_model(**kwargs):
    X_train_scaled = kwargs['ti'].xcom_pull(key='X_train_scaled')
    y_train = kwargs['ti'].xcom_pull(key='y_train')

    if X_train_scaled is None or y_train is None:
        raise ValueError("Training data not initialized. Make sure previous tasks are successful.")

    # Convert the lists back to arrays
    X_train_scaled = pd.DataFrame(X_train_scaled)
    y_train = pd.Series(y_train)

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
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='preprocess_dataset',
    python_callable=preprocess_dataset,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='scale_dataset',
    python_callable=scale_dataset,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='weather_model',
    python_callable=weather_model,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
t1 >> t2 >> t3 >> t4
