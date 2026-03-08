# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - gcc, g++: Required for building some Python packages
# - curl: Useful for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
# Docker will cache this layer if requirements.txt doesn't change
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create cache directory for persisted data
# This will store embeddings, clustering model, and cache
RUN mkdir -p /app/cache_data

# Expose port 8000
EXPOSE 8000

# Health check
# Checks if the API is responding every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn server
# - host 0.0.0.0: Listen on all interfaces (required for Docker)
# - port 8000: Standard HTTP port
# - app:app: module:application (src.app:app)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
