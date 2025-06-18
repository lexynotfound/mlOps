from flask import Flask, request, jsonify
import json
import datetime

app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive Grafana alerts"""
    try:
        alert_data = request.json
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("=" * 60)
        print(f"🚨 ALERT RECEIVED AT {timestamp}")
        print("=" * 60)

        if alert_data:
            # Parse Grafana alert
            alerts = alert_data.get('alerts', [])

            for alert in alerts:
                status = alert.get('status', 'unknown')
                labels = alert.get('labels', {})
                annotations = alert.get('annotations', {})

                print(f"📋 Alert Status: {status.upper()}")
                print(f"🏷️  Alert Name: {labels.get('alertname', 'Unknown')}")
                print(f"📊 Summary: {annotations.get('summary', 'No summary')}")
                print(f"📝 Description: {annotations.get('description', 'No description')}")
                print(f"🔗 Generator URL: {alert.get('generatorURL', 'N/A')}")

                if labels:
                    print("🏷️  Labels:")
                    for key, value in labels.items():
                        print(f"    {key}: {value}")

                print("-" * 40)

        # Save alert to file
        with open('alerts_received.json', 'a') as f:
            alert_record = {
                "timestamp": timestamp,
                "alert_data": alert_data
            }
            f.write(json.dumps(alert_record) + "\n")

        print("✅ Alert saved to alerts_received.json")
        print("=" * 60)

        return jsonify({"status": "success", "message": "Alert received"}), 200

    except Exception as e:
        print(f"❌ Error processing alert: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check for webhook server"""
    return jsonify({
        "status": "healthy",
        "service": "MLOps Webhook Server",
        "timestamp": datetime.datetime.now().isoformat()
    }), 200


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get received alerts"""
    try:
        with open('alerts_received.json', 'r') as f:
            alerts = [json.loads(line) for line in f]
        return jsonify({"alerts": alerts[-10:]}), 200  # Last 10 alerts
    except FileNotFoundError:
        return jsonify({"alerts": [], "message": "No alerts received yet"}), 200


if __name__ == '__main__':
    print("🔗 Starting MLOps Webhook Server...")
    print("📡 Listening for alerts on: http://localhost:5001/webhook")
    print("🏥 Health check: http://localhost:5001/health")
    print("📋 View alerts: http://localhost:5001/alerts")
    print("💾 Alerts will be saved to: alerts_received.json")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=False)