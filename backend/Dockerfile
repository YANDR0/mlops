FROM pytorch/pytorch:2.2.0-cpu

WORKDIR /app

# Install lightweight HTTP client for downloading model if needed
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps (if you add or change requirements)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and entrypoint
COPY main.py /app/main.py
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Model file can be mounted or downloaded at container start using MODEL_URL env var
VOLUME ["/app/model"]

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
