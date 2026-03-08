# Quick Start Guide

Get the Semantic Search API running in under 5 minutes.

## Option 1: Docker (Recommended) ⚡

**Fastest way to get started:**

```bash
# Clone or navigate to project directory
cd semantic-search

# Build and start (one command!)
docker-compose up --build
```

**That's it!** The API is now running at `http://localhost:8000`

**First run:** ~3 minutes (downloads dataset, builds embeddings)  
**Subsequent runs:** ~5 seconds (loads from cache)

### Test it:

```bash
# Query the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning"}'

# View interactive docs
open http://localhost:8000/docs
```

### Stop it:

```bash
docker-compose down
```

---

## Option 2: Local Python Environment

**If you prefer running Python directly:**

```bash
# 1. Setup virtual environment
bash setup_venv.sh
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 2. Start the API
uvicorn src.app:app --reload
```

**That's it!** The API is now running at `http://localhost:8000`

---

## What Happens on First Run?

1. **Downloads 20 Newsgroups dataset** (~30 seconds)
   - Automatically via sklearn
   - ~20,000 news posts
   - Stored in `~/.scikit_learn_data/`

2. **Builds embeddings** (~2 minutes)
   - Uses sentence-transformers
   - Generates 384-dim vectors
   - Stores in `cache_data/vector_store/`

3. **Fits clustering** (~30 seconds)
   - GMM with BIC selection
   - Discovers 8-12 semantic clusters
   - Stores in `cache_data/clustering/`

4. **Initializes cache** (<1 second)
   - Empty cache ready
   - Stores in `cache_data/cache/`

**Total:** ~3 minutes

**Subsequent runs:** ~5 seconds (loads from cache)

---

## API Endpoints

### POST /query - Semantic Search

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence"}'
```

**Response:**
```json
{
  "query": "artificial intelligence",
  "cache_hit": false,
  "result": {
    "documents": [...],
    "cluster_distribution": {"3": 0.45, "7": 0.35}
  },
  "dominant_cluster": 3
}
```

### GET /cache/stats - Cache Metrics

```bash
curl "http://localhost:8000/cache/stats"
```

**Response:**
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache - Clear Cache

```bash
curl -X DELETE "http://localhost:8000/cache"
```

### GET /health - Health Check

```bash
curl "http://localhost:8000/health"
```

---

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test all endpoints directly in the browser!

---

## Example Queries

Try these queries to see the system in action:

```bash
# Technology
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "computer programming"}'

# Science
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration"}'

# Sports
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "baseball statistics"}'

# Similar query (should hit cache)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms"}'

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning neural networks"}'
```

---

## Analyze Clustering

Explore the semantic structure discovered by the system:

```bash
# Local
python -m src.analyze_clusters

# Docker
docker-compose exec semantic-search python -m src.analyze_clusters
```

This shows:
- Cluster composition (what topics are in each cluster)
- Boundary cases (documents spanning multiple clusters)
- Uncertainty analysis (where the model is unsure)
- Soft vs hard clustering comparison

---

## Troubleshooting

### Port 8000 already in use

```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or use different port
uvicorn src.app:app --port 8001
```

### Module not found errors

```bash
# Make sure you're in the project root
pwd  # Should show .../semantic-search

# Reinstall dependencies
pip install -r requirements.txt
```

### Docker build fails

```bash
# Clean build (no cache)
docker-compose build --no-cache

# Check Docker is running
docker ps
```

### Slow first run

This is normal! The system:
- Downloads dataset (~30s)
- Builds embeddings (~2min)
- Fits clustering (~30s)

Subsequent runs are fast (~5s).

---

## Next Steps

1. **Read the full documentation:** [README.md](README.md)
2. **Learn about Docker deployment:** [DOCKER.md](DOCKER.md)
3. **Understand design decisions:** See README.md sections
4. **Explore the code:** Check `src/` directory
5. **Run tests:** `python -m tests.test_system`

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `docker-compose up --build` | Start with Docker |
| `uvicorn src.app:app --reload` | Start with Python |
| `curl http://localhost:8000/docs` | View API docs |
| `curl http://localhost:8000/health` | Health check |
| `docker-compose down` | Stop Docker |
| `python -m src.analyze_clusters` | Analyze clustering |
| `python -m tests.test_system` | Run tests |

---

**Questions?** Check [README.md](README.md) for comprehensive documentation.

**Issues?** Make sure you're in the project root and dependencies are installed.

**Ready to deploy?** See [DOCKER.md](DOCKER.md) for production deployment.
