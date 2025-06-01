import os
import time
import pandas as pd
import numpy as np
import traceback
from flask import Flask, request, jsonify
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inference_service")

# Define Prometheus metrics
PREDICTIONS_COUNTER = Counter('api_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('api_prediction_latency_seconds', 'Prediction latency in seconds')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
MODEL_CONFIDENCE = Gauge('api_confidence_score', 'Confidence score for predictions')

# Start Prometheus metrics server
try:
    start_http_server(8001)  # Use different port from Flask app
    logger.info("Prometheus metrics server started on port 8001")
except Exception as e:
    logger.warning(f"Could not start Prometheus server: {str(e)}")


# Dummy model implementation - completely manual
class DummyModel:
    """Dummy clustering model that simulates KMeans"""

    def __init__(self):
        # Define 3 clusters based on salary
        self.cluster_centers_ = np.array([
            [5000000],  # Low salary
            [10000000],  # Medium salary
            [15000000]  # High salary
        ])

    def predict(self, X):
        """Predict closest cluster based on salary"""
        # Find the closest cluster center
        distances = np.abs(X - self.cluster_centers_)
        return np.argmin(distances, axis=0)


class DummyVectorizer:
    """Dummy text vectorizer"""

    def transform(self, texts):
        """Transform text into a simple feature vector"""
        # Create dummy vector that will work with similarity calculation
        return np.ones((1, 1))


class DummyPreprocessor:
    """Dummy preprocessor"""

    def transform(self, X):
        """Extract salary feature from input data"""
        if isinstance(X, pd.DataFrame) and 'Expected salary (IDR)' in X.columns:
            try:
                salary = float(X['Expected salary (IDR)'].iloc[0]) if len(X) > 0 else 5000000
            except (ValueError, TypeError):
                salary = 5000000
        else:
            salary = 5000000

        return np.array([[salary]])


# Initialize with dummy components
model = DummyModel()
vectorizer = DummyVectorizer()
preprocessor = DummyPreprocessor()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    API_REQUESTS.labels(endpoint='/health', status='success').inc()
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "components": {
            "model_type": "DummyModel",
            "vectorizer_type": "DummyVectorizer",
            "preprocessor_type": "DummyPreprocessor"
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions"""
    start_time = time.time()

    try:
        # Get data from request
        data = request.json

        if not data or not isinstance(data, dict):
            API_REQUESTS.labels(endpoint='/predict', status='error').inc()
            return jsonify({"error": "Invalid input format"}), 400

        # Extract candidate information
        candidate_info = {
            'Full name': data.get('name', ''),
            'Gender': data.get('gender', ''),
            'Marital status': data.get('marital_status', ''),
            'Highest formal of education': data.get('education', ''),
            'Faculty/Major': data.get('major', ''),
            'Current status': data.get('current_status', ''),
            'Experience': data.get('experience', ''),
            'Expected salary (IDR)': data.get('expected_salary', 0)
        }

        # Create DataFrame
        df = pd.DataFrame([candidate_info])

        # Preprocess input data
        features = preprocessor.transform(df)

        # Make prediction (cluster assignment)
        cluster = model.predict(features)[0]

        # Process job information if provided
        job_fit_scores = {}
        if 'job_description' in data:
            # For demo purposes, generate a confidence score based on education level
            edu_map = {
                'High School': 0.5,
                'Diploma': 0.6,
                'Bachelor': 0.7,
                'Master': 0.8,
                'PhD': 0.9
            }
            education = data.get('education', 'Bachelor')
            confidence = edu_map.get(education, 0.7)

            # Add some randomness (±0.1)
            confidence = min(max(confidence + np.random.uniform(-0.1, 0.1), 0.5), 0.95)

            # Update metrics
            MODEL_CONFIDENCE.set(confidence)

            job_fit_scores = {
                "job_title": data.get('job_title', 'Unspecified Position'),
                "cluster_id": int(cluster),
                "confidence_score": float(confidence),
                "recommendation": "Recommended" if confidence > 0.7 else "Consider" if confidence > 0.5 else "Not Recommended"
            }
        else:
            # Just return cluster without job matching
            job_fit_scores = {
                "cluster_id": int(cluster)
            }

        # Record metrics
        PREDICTIONS_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        API_REQUESTS.labels(endpoint='/predict', status='success').inc()

        # Return prediction results
        return jsonify({
            "prediction": job_fit_scores,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        API_REQUESTS.labels(endpoint='/predict', status='error').inc()
        return jsonify({"error": str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    start_time = time.time()

    try:
        # Get data from request
        data = request.json

        if not data or not isinstance(data, list):
            API_REQUESTS.labels(endpoint='/batch_predict', status='error').inc()
            return jsonify({"error": "Input must be a list of candidates"}), 400

        results = []

        # Process each candidate
        for candidate in data:
            # Extract candidate information
            candidate_info = {
                'Full name': candidate.get('name', ''),
                'Gender': candidate.get('gender', ''),
                'Marital status': candidate.get('marital_status', ''),
                'Highest formal of education': candidate.get('education', ''),
                'Faculty/Major': candidate.get('major', ''),
                'Current status': candidate.get('current_status', ''),
                'Experience': candidate.get('experience', ''),
                'Expected salary (IDR)': candidate.get('expected_salary', 0)
            }

            # Create DataFrame
            df = pd.DataFrame([candidate_info])

            # Preprocess input data
            features = preprocessor.transform(df)

            # Make prediction (cluster assignment)
            cluster = model.predict(features)[0]

            # Add to results
            results.append({
                "candidate_id": candidate.get('id', 'unknown'),
                "cluster_id": int(cluster)
            })

        # Record metrics
        batch_size = len(data)
        PREDICTIONS_COUNTER.inc(batch_size)
        PREDICTION_LATENCY.observe(time.time() - start_time)
        API_REQUESTS.labels(endpoint='/batch_predict', status='success').inc()

        # Return prediction results
        return jsonify({
            "predictions": results,
            "count": batch_size,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        API_REQUESTS.labels(endpoint='/batch_predict', status='error').inc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)