import os
import time
import json
import numpy as np
import traceback
from flask import Flask, request, jsonify
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
import threading

# Configure Flask app
app = Flask(__name__)

# Configure logging with ASCII-safe format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_service.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference_service")

# PROMETHEUS METRICS FOR GRAFANA
PREDICTIONS_COUNTER = Counter('ml_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Prediction latency in seconds')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
MODEL_CONFIDENCE = Gauge('ml_model_confidence_score', 'Model confidence score')
ACTIVE_USERS = Gauge('api_active_users', 'Number of active users')
ERROR_RATE = Gauge('api_error_rate', 'API error rate percentage')
MEMORY_USAGE = Gauge('api_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('api_cpu_usage_percent', 'CPU usage percentage')

# Start Prometheus metrics server
try:
    start_http_server(8001)  # Port 8001 for metrics
    logger.info("Prometheus metrics server started on port 8001")
except Exception as e:
    logger.warning(f"Could not start Prometheus server: {str(e)}")


# Dummy model for demo (removed pandas dependency)
class DummyModel:
    def __init__(self):
        self.clusters = 3
        self.confidence_threshold = 0.7

    def predict(self, features):
        # Simulate cluster prediction
        cluster = np.random.randint(0, self.clusters)
        confidence = np.random.uniform(0.5, 0.95)
        return cluster, confidence


# Initialize model
model = DummyModel()
logger.info("Model loaded successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        API_REQUESTS.labels(endpoint='/health', status='success').inc()

        # Update system metrics
        CPU_USAGE.set(np.random.uniform(10, 40))
        MEMORY_USAGE.set(np.random.uniform(100_000_000, 500_000_000))

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": True,
            "metrics_enabled": True,
            "uptime_seconds": time.time(),
            "version": "1.0.0"
        }

        logger.info("Health check successful")
        return jsonify(health_status), 200

    except Exception as e:
        API_REQUESTS.labels(endpoint='/health', status='error').inc()
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    start_time = time.time()

    try:
        # Get request data
        data = request.json
        if not data:
            API_REQUESTS.labels(endpoint='/predict', status='error').inc()
            return jsonify({"error": "No input data provided"}), 400

        # Extract features
        features = {
            'name': data.get('name', 'Unknown'),
            'experience': data.get('experience', 'Fresh Graduate'),
            'education': data.get('education', 'Bachelor'),
            'expected_salary': data.get('expected_salary', 5000000),
            'position': data.get('desired_position', 'Software Engineer')
        }

        # Make prediction
        cluster, confidence = model.predict(features)

        # Determine recommendation
        if confidence > 0.8:
            recommendation = "Highly Recommended"
            priority = "HIGH"
        elif confidence > 0.6:
            recommendation = "Recommended"
            priority = "MEDIUM"
        else:
            recommendation = "Consider"
            priority = "LOW"

        # Prepare response
        prediction_result = {
            "candidate_name": features['name'],
            "cluster_id": int(cluster),
            "confidence_score": float(confidence),
            "recommendation": recommendation,
            "priority": priority,
            "match_percentage": round(confidence * 100, 2),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }

        # Update metrics
        PREDICTIONS_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        MODEL_CONFIDENCE.set(confidence)
        API_REQUESTS.labels(endpoint='/predict', status='success').inc()
        ACTIVE_USERS.set(np.random.randint(1, 10))
        ERROR_RATE.set(np.random.uniform(0, 5))  # 0-5% error rate

        logger.info(f"Prediction successful: {prediction_result['candidate_name']} -> Cluster {cluster}")
        return jsonify(prediction_result), 200

    except Exception as e:
        API_REQUESTS.labels(endpoint='/predict', status='error').inc()
        ERROR_RATE.set(np.random.uniform(10, 25))  # Higher error rate
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    start_time = time.time()

    try:
        data = request.json
        if not isinstance(data, list):
            API_REQUESTS.labels(endpoint='/batch_predict', status='error').inc()
            return jsonify({"error": "Input must be a list of candidates"}), 400

        results = []
        for candidate in data:
            cluster, confidence = model.predict(candidate)
            results.append({
                "candidate_id": candidate.get('id', 'unknown'),
                "cluster_id": int(cluster),
                "confidence_score": float(confidence)
            })

        # Update metrics
        PREDICTIONS_COUNTER.inc(len(data))
        PREDICTION_LATENCY.observe(time.time() - start_time)
        API_REQUESTS.labels(endpoint='/batch_predict', status='success').inc()

        response = {
            "predictions": results,
            "total_candidates": len(data),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Batch prediction completed: {len(data)} candidates")
        return jsonify(response), 200

    except Exception as e:
        API_REQUESTS.labels(endpoint='/batch_predict', status='error').inc()
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    try:
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        return str(e), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """API statistics endpoint"""
    try:
        stats = {
            "total_predictions": int(PREDICTIONS_COUNTER._value._value),
            "current_confidence": MODEL_CONFIDENCE._value._value,
            "active_users": int(ACTIVE_USERS._value._value),
            "error_rate_percent": ERROR_RATE._value._value,
            "cpu_usage_percent": CPU_USAGE._value._value,
            "memory_usage_mb": MEMORY_USAGE._value._value / 1_000_000,
            "uptime_seconds": time.time(),
            "status": "operational"
        }

        API_REQUESTS.labels(endpoint='/stats', status='success').inc()
        return jsonify(stats), 200

    except Exception as e:
        API_REQUESTS.labels(endpoint='/stats', status='error').inc()
        logger.error(f"Stats endpoint failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


def simulate_background_activity():
    """Background activity to generate realistic metrics"""
    def background_task():
        while True:
            try:
                # Simulate varying load
                CPU_USAGE.set(np.random.uniform(5, 45))
                MEMORY_USAGE.set(np.random.uniform(80_000_000, 600_000_000))
                ACTIVE_USERS.set(np.random.randint(1, 15))
                ERROR_RATE.set(np.random.uniform(0, 8))

                # Random confidence variations
                MODEL_CONFIDENCE.set(np.random.uniform(0.6, 0.95))

                time.sleep(30)  # Update every 30 seconds
            except:
                pass

    thread = threading.Thread(target=background_task, daemon=True)
    thread.start()


if __name__ == '__main__':
    # Start background metrics simulation
    simulate_background_activity()

    print("Starting Inference Service...")
    print("Endpoints:")
    print("   - Health Check: http://localhost:3000/health")
    print("   - Prediction: http://localhost:3000/predict")
    print("   - Batch Predict: http://localhost:3000/batch_predict")
    print("   - Statistics: http://localhost:3000/stats")
    print("   - Metrics: http://localhost:3000/metrics")
    print("Prometheus Metrics: http://localhost:8001")
    print("Ready for Grafana monitoring!")

    # Start Flask app
    app.run(host='0.0.0.0', port=3000, debug=False)