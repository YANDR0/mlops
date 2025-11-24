#!/usr/bin/env bash
set -e

# If MODEL_URL is provided and model not present, download it into /app
MODEL_PATH="/app/best_model.pth"
if [ -n "${MODEL_URL}" ] && [ ! -f "${MODEL_PATH}" ]; then
  echo "MODEL_URL provided; downloading model..."
  curl -fsSL "$MODEL_URL" -o "$MODEL_PATH"
  echo "Model downloaded to $MODEL_PATH"
fi

# If model is mounted at /app/model/best_model.pth, copy it into place
if [ -f "/app/model/best_model.pth" ] && [ ! -f "${MODEL_PATH}" ]; then
  echo "Found mounted model at /app/model/best_model.pth, copying into app dir"
  cp /app/model/best_model.pth "$MODEL_PATH"
fi

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000
