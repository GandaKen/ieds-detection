# Dockerfile


FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    pip install python-multipart && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY FastAPI.py ./

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
