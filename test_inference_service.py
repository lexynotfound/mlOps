import requests
import time
import json


def test_health_endpoint():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:3000/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_predict_endpoint():
    """Test prediction endpoint with sample data"""
    print("\nTesting prediction endpoint...")

    test_cases = [
        {
            "name": "John Doe",
            "experience": "Senior",
            "education": "Master",
            "expected_salary": 8000000,
            "desired_position": "ML Engineer"
        },
        {
            "name": "Jane Smith",
            "experience": "Fresh Graduate",
            "education": "Bachelor",
            "expected_salary": 5000000,
            "desired_position": "Data Analyst"
        },
        {
            "name": "Bob Wilson",
            "experience": "Mid-level",
            "education": "Bachelor",
            "expected_salary": 7000000,
            "desired_position": "Software Engineer"
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        try:
            print(f"Testing case {i + 1}: {test_case['name']}")
            response = requests.post(
                "http://localhost:3000/predict",
                json=test_case,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                results.append(result)
                print(f"  Cluster: {result['cluster_id']}")
                print(f"  Confidence: {result['confidence_score']:.3f}")
                print(f"  Recommendation: {result['recommendation']}")
                print(f"  Processing time: {result['processing_time_ms']}ms")
            else:
                print(f"  Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"  Request failed: {e}")

        time.sleep(1)  # Small delay between requests

    return results


def test_batch_predict():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")

    batch_data = [
        {"id": "candidate_1", "name": "Alice", "experience": "Senior"},
        {"id": "candidate_2", "name": "Bob", "experience": "Mid-level"},
        {"id": "candidate_3", "name": "Charlie", "experience": "Fresh Graduate"}
    ]

    try:
        response = requests.post(
            "http://localhost:3000/batch_predict",
            json=batch_data,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Batch processed: {result['total_candidates']} candidates")
            print(f"Processing time: {result['processing_time_ms']}ms")
            for pred in result['predictions']:
                print(
                    f"  {pred['candidate_id']}: Cluster {pred['cluster_id']} (confidence: {pred['confidence_score']:.3f})")
            return result
        else:
            print(f"Batch prediction failed: {response.status_code}")

    except Exception as e:
        print(f"Batch prediction error: {e}")

    return None


def test_stats_endpoint():
    """Test statistics endpoint"""
    print("\nTesting stats endpoint...")

    try:
        response = requests.get("http://localhost:3000/stats")
        if response.status_code == 200:
            stats = response.json()
            print("Current Statistics:")
            print(f"  Total predictions: {stats['total_predictions']}")
            print(f"  Active users: {stats['active_users']}")
            print(f"  Error rate: {stats['error_rate_percent']:.2f}%")
            print(f"  CPU usage: {stats['cpu_usage_percent']:.2f}%")
            print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
            print(f"  Current confidence: {stats['current_confidence']:.3f}")
            return stats
        else:
            print(f"Stats request failed: {response.status_code}")

    except Exception as e:
        print(f"Stats request error: {e}")

    return None


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    print("\nTesting metrics endpoint...")

    try:
        response = requests.get("http://localhost:3000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            print("Metrics endpoint working - sample metrics:")

            # Extract some key metrics
            lines = metrics_text.split('\n')
            for line in lines[:20]:  # Show first 20 lines
                if line and not line.startswith('#'):
                    print(f"  {line}")

            print("  ... (more metrics available)")
            return True
        else:
            print(f"Metrics request failed: {response.status_code}")

    except Exception as e:
        print(f"Metrics request error: {e}")

    return False


def load_test():
    """Generate load to trigger alerts"""
    print("\nGenerating load for alerting test...")

    for i in range(10):
        try:
            response = requests.post(
                "http://localhost:3000/predict",
                json={"name": f"LoadTest_{i}", "experience": "Mid-level"},
                timeout=2
            )
            print(f"Load test {i + 1}/10: {response.status_code}")

        except Exception as e:
            print(f"Load test {i + 1}/10: Failed - {e}")

        time.sleep(0.5)  # Fast requests to generate metrics


def check_prometheus_targets():
    """Check if Prometheus can scrape our service"""
    print("\nChecking Prometheus targets...")

    try:
        response = requests.get("http://localhost:9090/api/v1/targets")
        if response.status_code == 200:
            targets = response.json()
            print("Prometheus targets status:")

            for target in targets.get('data', {}).get('activeTargets', []):
                job = target.get('labels', {}).get('job', 'unknown')
                health = target.get('health', 'unknown')
                endpoint = target.get('scrapeUrl', 'unknown')
                print(f"  Job: {job} | Health: {health} | URL: {endpoint}")

            return True
        else:
            print(f"Prometheus targets check failed: {response.status_code}")

    except Exception as e:
        print(f"Prometheus not accessible: {e}")

    return False


def save_test_evidence():
    """Save comprehensive test evidence"""
    evidence = {
        "test_timestamp": time.time(),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "services_tested": {
            "inference_service": "http://localhost:3000",
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3001"
        },
        "test_results": {}
    }

    # Test all endpoints
    print("=" * 60)
    print("COMPREHENSIVE SERVICE TESTING")
    print("=" * 60)

    # Health check
    health_ok = test_health_endpoint()
    evidence["test_results"]["health_check"] = health_ok

    # Prediction tests
    prediction_results = test_predict_endpoint()
    evidence["test_results"]["predictions"] = prediction_results

    # Batch prediction
    batch_results = test_batch_predict()
    evidence["test_results"]["batch_prediction"] = batch_results

    # Stats
    stats = test_stats_endpoint()
    evidence["test_results"]["statistics"] = stats

    # Metrics
    metrics_ok = test_metrics_endpoint()
    evidence["test_results"]["metrics_available"] = metrics_ok

    # Load test
    load_test()
    evidence["test_results"]["load_test_completed"] = True

    # Prometheus check
    prometheus_ok = check_prometheus_targets()
    evidence["test_results"]["prometheus_scraping"] = prometheus_ok

    # Save evidence
    with open('service_test_evidence.json', 'w') as f:
        json.dump(evidence, f, indent=2)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Health Check: {'✓' if health_ok else '✗'}")
    print(f"Predictions: {'✓' if prediction_results else '✗'}")
    print(f"Batch Prediction: {'✓' if batch_results else '✗'}")
    print(f"Statistics: {'✓' if stats else '✗'}")
    print(f"Metrics: {'✓' if metrics_ok else '✗'}")
    print(f"Prometheus: {'✓' if prometheus_ok else '✗'}")

    print(f"\nEvidence saved to: service_test_evidence.json")

    print("\nREADY FOR SUBMISSION:")
    print("1. Take screenshot of Grafana dashboard")
    print("2. Take screenshot of API responses")
    print("3. Take screenshot of Prometheus targets")
    print("4. Submit service_test_evidence.json")


if __name__ == "__main__":
    save_test_evidence()