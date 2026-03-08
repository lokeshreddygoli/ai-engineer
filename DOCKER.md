# Docker Deployment Guide

Complete guide for containerizing and deploying the Semantic Search API.

## Quick Start

```bash
# Build and run with docker-compose (recommended)
docker-compose up --build

# Or using Docker directly
docker build -t semantic-search .
docker run -p 8000:8000 -v $(pwd)/cache_data:/app/cache_data semantic-search
```

API available at: `http://localhost:8000`

---

## Dockerfile Explanation

### Base Image Choice

```dockerfile
FROM python:3.11-slim
```

**Why python:3.11-slim?**
- Size: ~150MB (vs 1GB for full Python image)
- Security: Fewer packages = smaller attack surface
- Performance: Latest Python with performance improvements
- Trade-off: Need to install build tools (gcc, g++)

**Alternatives considered:**
| Image | Size | Pros | Cons | Decision |
|-------|------|------|------|----------|
| python:3.11-slim | 150MB | Small, secure | Need build tools | ✓ Chosen |
| python:3.11 | 1GB | All tools included | Too large | Rejected |
| python:3.11-alpine | 50MB | Smallest | Compatibility issues | Rejected |

### Layer Optimization

```dockerfile
# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code later
COPY src/ ./src/
```

**Why this order?**
- Docker caches layers
- requirements.txt changes less frequently than code
- Result: Faster rebuilds when only code changes

**Example:**
- Code change only: ~5 seconds (reuses dependency layer)
- Dependency change: ~2 minutes (rebuilds from requirements)

### Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**Parameters explained:**
- `interval=30s`: Check every 30 seconds
- `timeout=10s`: Fail if check takes >10s
- `start-period=180s`: Wait 3 minutes before first check (dataset download)
- `retries=3`: Mark unhealthy after 3 failures

**Why 180s start period?**
- First run downloads 20 Newsgroups dataset (~30s)
- Builds embeddings (~2 minutes)
- Total: ~3 minutes
- Subsequent runs: ~5 seconds (loads from cache)

### Volume Mounting

```dockerfile
RUN mkdir -p /app/cache_data
```

**Why mount cache_data?**
- Persists embeddings, clustering model, cache
- Avoids rebuilding on container restart
- Result: 5s restart vs 3min rebuild

---

## docker-compose.yml Explanation

### Service Configuration

```yaml
services:
  semantic-search:
    build:
      context: .
      dockerfile: Dockerfile
```

**Why docker-compose?**
- Simpler commands (`docker-compose up` vs long `docker run`)
- Declarative configuration
- Easy to add services (e.g., Redis, monitoring)
- Environment management

### Volume Mounting

```yaml
volumes:
  - ./cache_data:/app/cache_data
```

**What this does:**
- Maps host `./cache_data` to container `/app/cache_data`
- Data persists between container restarts
- Can inspect/backup data on host

**Example:**
```bash
# First run
docker-compose up  # Downloads dataset, builds embeddings (3 min)
docker-compose down

# Second run
docker-compose up  # Loads from cache (5 sec)
```

### Restart Policy

```yaml
restart: unless-stopped
```

**Options:**
- `no`: Never restart (default)
- `always`: Always restart
- `on-failure`: Restart only on error
- `unless-stopped`: Restart unless manually stopped (✓ chosen)

**Why unless-stopped?**
- Survives host reboots
- Doesn't restart if manually stopped
- Good for production

---

## Docker Commands

### Building

```bash
# Build image
docker build -t semantic-search .

# Build with no cache (clean build)
docker build --no-cache -t semantic-search .

# Build with docker-compose
docker-compose build

# Build and start
docker-compose up --build
```

### Running

```bash
# Run in foreground
docker run -p 8000:8000 semantic-search

# Run in background
docker run -d -p 8000:8000 semantic-search

# Run with volume mounting
docker run -p 8000:8000 \
  -v $(pwd)/cache_data:/app/cache_data \
  semantic-search

# Using docker-compose
docker-compose up        # Foreground
docker-compose up -d     # Background
```

### Managing

```bash
# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart container
docker-compose restart

# Execute command in running container
docker-compose exec semantic-search python -m src.analyze_clusters
```

### Debugging

```bash
# Shell into running container
docker-compose exec semantic-search bash

# View container status
docker-compose ps

# View resource usage
docker stats

# Inspect container
docker inspect semantic-search-api
```

---

## Production Deployment

### AWS ECS (Elastic Container Service)

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name semantic-search

# 2. Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag image
docker tag semantic-search:latest \
  <account>.dkr.ecr.us-east-1.amazonaws.com/semantic-search:latest

# 4. Push image
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/semantic-search:latest

# 5. Create ECS task definition and service (via AWS Console or CLI)
```

### Google Cloud Run

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/<project-id>/semantic-search

# 2. Deploy to Cloud Run
gcloud run deploy semantic-search \
  --image gcr.io/<project-id>/semantic-search \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --memory 2Gi \
  --timeout 300s \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# 1. Create resource group
az group create --name semantic-search-rg --location eastus

# 2. Create container registry
az acr create --resource-group semantic-search-rg \
  --name semanticsearchacr --sku Basic

# 3. Login to registry
az acr login --name semanticsearchacr

# 4. Tag and push image
docker tag semantic-search semanticsearchacr.azurecr.io/semantic-search:latest
docker push semanticsearchacr.azurecr.io/semantic-search:latest

# 5. Deploy container
az container create \
  --resource-group semantic-search-rg \
  --name semantic-search \
  --image semanticsearchacr.azurecr.io/semantic-search:latest \
  --dns-name-label semantic-search \
  --ports 8000
```

### DigitalOcean App Platform

```bash
# 1. Push to Docker Hub
docker tag semantic-search <username>/semantic-search:latest
docker push <username>/semantic-search:latest

# 2. Create app via DigitalOcean Console
# - Select Docker Hub as source
# - Set port to 8000
# - Configure health check: /health
```

---

## Performance Optimization

### Multi-Stage Build (Optional)

For smaller images, use multi-stage build:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Result:** ~100MB smaller image

### Caching Strategy

```bash
# Mount cache as volume for persistence
docker run -p 8000:8000 \
  -v semantic-search-cache:/app/cache_data \
  semantic-search

# Or use named volume in docker-compose
volumes:
  semantic-search-cache:
    driver: local
```

---

## Troubleshooting

### Container exits immediately

```bash
# Check logs
docker-compose logs

# Common causes:
# - Port 8000 already in use
# - Missing dependencies
# - Syntax error in code
```

### Health check failing

```bash
# Check health status
docker inspect semantic-search-api | grep Health

# Common causes:
# - First run (wait 3 minutes for dataset download)
# - API crashed (check logs)
# - Port not exposed
```

### Volume mounting not working

```bash
# Check volume
docker volume ls
docker volume inspect <volume-name>

# Verify mount
docker-compose exec semantic-search ls -la /app/cache_data
```

### Out of memory

```bash
# Increase memory limit
docker run -p 8000:8000 --memory=2g semantic-search

# Or in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

---

## Security Best Practices

### 1. Non-root User

Add to Dockerfile:

```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Read-only Filesystem

```bash
docker run -p 8000:8000 --read-only \
  -v $(pwd)/cache_data:/app/cache_data \
  semantic-search
```

### 3. Security Scanning

```bash
# Scan image for vulnerabilities
docker scan semantic-search

# Or use Trivy
trivy image semantic-search
```

### 4. Secrets Management

```bash
# Use environment variables
docker run -p 8000:8000 \
  -e API_KEY=secret \
  semantic-search

# Or use Docker secrets (Swarm mode)
echo "secret" | docker secret create api_key -
```

---

## Monitoring

### Health Checks

```bash
# Manual health check
curl http://localhost:8000/health

# Automated monitoring
while true; do
  curl -f http://localhost:8000/health || echo "UNHEALTHY"
  sleep 30
done
```

### Logs

```bash
# View logs
docker-compose logs -f

# Export logs
docker-compose logs > logs.txt

# Log rotation (in docker-compose.yml)
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Metrics

```bash
# Resource usage
docker stats semantic-search-api

# Detailed metrics
docker inspect semantic-search-api | jq '.[0].State'
```

---

## Summary

**Docker provides:**
- ✓ Consistent environment
- ✓ Easy deployment
- ✓ Automatic health checks
- ✓ Data persistence via volumes
- ✓ Production-ready configuration

**Quick commands:**
```bash
# Development
docker-compose up --build

# Production
docker-compose up -d

# Monitoring
docker-compose logs -f

# Cleanup
docker-compose down -v
```

**Image size:** ~800MB
**Startup time:** 3 min (first run), 5 sec (subsequent)
**Memory usage:** ~500MB baseline
