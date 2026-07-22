FROM python:3.10-slim

# ffmpeg needed for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY src/ ./src/
COPY models/ ./models/
COPY frontend/ ./frontend/

# HuggingFace uses a fixed port (7860); Render provides $PORT — support both.
EXPOSE 7860
ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-7860}"]
