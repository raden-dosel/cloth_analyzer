# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements/prod.txt .
RUN pip install --no-cache-dir -r prod.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Startup command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]