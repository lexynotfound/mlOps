# Gunakan image Python
FROM python:3.12.7-slim
LABEL authors="raiha"

# Set working directory
WORKDIR /app

# Salin requirements.txt terlebih dahulu (untuk better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Expose port untuk Flask dan Prometheus
EXPOSE 3000
EXPOSE 8001

# Jalankan Prometheus exporter dan inference API secara bersamaan
CMD ["sh", "-c", "python monitor/prometheus/prometheus_exporter.py & python inference.py"]