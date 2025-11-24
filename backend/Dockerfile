FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app


# Install system deps and lightweight HTTP client for downloading model if needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	ca-certificates \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps. Install PyTorch CPU wheels from official index.
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Install torch + torchvision CPU wheels from the PyTorch index then install other requirements
RUN pip install --no-cache-dir "torch==2.2.0+cpu" "torchvision==0.15.2+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html || \
	pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and entrypoint
COPY main.py /app/main.py
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Model file can be mounted or downloaded at container start using MODEL_URL env var
VOLUME ["/app/model"]

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
