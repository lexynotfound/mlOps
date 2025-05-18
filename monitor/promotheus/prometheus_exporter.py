from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import time
import random
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_metrics")

# Define Prometheus metrics
PREDICTIONS = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Histogram of prediction latency')
CONFIDENCE_SCORE = Gauge('model_confidence_score', 'Confidence score of predictions')
CLUSTER_DISTRIBUTION = Gauge('model_cluster_distribution', 'Distribution of samples across clusters', ['cluster'])
PREDICTION_ERRORS = Counter('model_prediction_errors', 'Prediction errors', ['error_type'])
CPU_USAGE = Gauge('model_cpu_usage_percent', 'CPU usage percentage of the model')
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage in bytes')
DATA_DRIFT = Gauge('model_data_drift_score', 'Data drift detection score')
FEATURE_IMPORTANCE = Gauge('model_feature_importance', 'Feature importance values', ['feature'])
BATCH_SIZE = Summary('model_batch_size', 'Summary of batch sizes for predictions')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load model in seconds')
REQUEST_RATE = Counter('request_rate', 'Rate of requests per second')
MODEL_VERSION = Gauge('model_version', 'Model version information', ['version'])

# Mock data for demonstration
FEATURES = ['education_score', 'experience_years', 'salary_expectation', 'skill_match']


# Load model for simulations
def load_model():
    """Load model for metrics simulation"""
    start_time = time.time()
    try:
        # In a real scenario, you would load your actual model
        # model = mlflow.sklearn.load_model("mlflow-artifacts/kmeans_model")
        # For simulation, we'll create a dummy model structure
        model = {
            'n_clusters': 5,
            'version': '1.0.0',
            'file_path': '/app/models/candidate_recommender.pkl'
        }
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        MODEL_VERSION.labels(version='1.0.0').set(1)
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return model
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type='model_loading').inc()
        logger.error(f"Error loading model: {str(e)}")
        return None


# Simulate model predictions and metrics updates
def simulate_prediction(model, batch_size=1):
    """Simulate model prediction and update metrics"""
    start_time = time.time()

    try:
        # Simulate prediction process
        REQUEST_RATE.inc()
        BATCH_SIZE.observe(batch_size)

        # Simulate processing time based on batch size
        processing_time = 0.01 * batch_size + random.uniform(0.01, 0.05)
        time.sleep(processing_time)

        # Simulate cluster assignment (in real case would be model.predict())
        cluster_id = random.randint(0, model['n_clusters'] - 1)

        # Update metrics
        PREDICTIONS.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        confidence = random.uniform(0.6, 0.99)
        CONFIDENCE_SCORE.set(confidence)
        CLUSTER_DISTRIBUTION.labels(cluster=str(cluster_id)).inc()

        # Simulate system resource usage
        CPU_USAGE.set(random.uniform(10, 40))
        MEMORY_USAGE.set(random.uniform(100_000_000, 500_000_000))

        # Simulate data drift score (0-1, higher means more drift)
        drift_score = random.betavariate(1.2, 5)  # Skewed toward lower values
        DATA_DRIFT.set(drift_score)

        # Simulate feature importance values
        for feature in FEATURES:
            FEATURE_IMPORTANCE.labels(feature=feature).set(random.uniform(0, 1))

        # Simulate API request tracking
        status = 'success' if random.random() > 0.05 else 'error'
        API_REQUESTS.labels(endpoint='/predict', status=status).inc()

        # Log simulation results
        logger.info(f"Prediction completed: cluster={cluster_id}, confidence={confidence:.4f}, drift={drift_score:.4f}")

        return {
            'cluster': cluster_id,
            'confidence': confidence,
            'processing_time': processing_time
        }

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type='prediction').inc()
        logger.error(f"Error during prediction: {str(e)}")
        return None


# Main monitoring function
def main():
    # Start up the server to expose the metrics
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    # Load model
    model = load_model()
    if not model:
        logger.error("Failed to load model. Exiting.")
        return

    # Simulate predictions and update metrics
    while True:
        # Randomly vary batch size to simulate different loads
        batch_size = np.random.poisson(3) + 1  # Poisson distribution with mean 3
        result = simulate_prediction(model, batch_size)

        # Simulate occasional errors
        if random.random() < 0.02:  # 2% chance of error
            error_type = random.choice(['timeout', 'input_validation', 'preprocessing'])
            PREDICTION_ERRORS.labels(error_type=error_type).inc()
            logger.warning(f"Simulated error occurred: {error_type}")

        # Wait before the next prediction
        time.sleep(random.uniform(0.2, 2.0))


if __name__ == "__main__":
    main()