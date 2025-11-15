# Docker Deployment Guide

## Quick Start

### Build and Run Locally
```bash
# Build the image
docker build -t ml-service:latest .

# Run the container
docker run -p 8000:8000 ml-service:latest

# Test the API
curl http://localhost:8000/health
```

### Using Make Commands
```bash
make docker_build    # Build image
make docker_run      # Run container
make docker_compose  # Build and run with docker-compose
```

## Docker Hub Publishing

### 1. Login to Docker Hub
```bash
docker login
```

### 2. Use the Publishing Script
```bash
./docker-publish.sh <your-username> v1.0.0
```

### 3. Manual Publishing
```bash
# Tag the image
docker tag ml-service:latest <username>/energy-efficiency-api:latest
docker tag ml-service:latest <username>/energy-efficiency-api:v1.0.0

# Push to Docker Hub
docker push <username>/energy-efficiency-api:latest
docker push <username>/energy-efficiency-api:v1.0.0
```

## Container Specifications

### Image Details
- **Base**: python:3.13-slim
- **Size**: ~200MB (optimized)
- **Port**: 8000
- **Working Dir**: /app

### Included Components
- FastAPI application
- All trained MLflow models
- Pinned dependencies
- Uvicorn ASGI server

### Environment Variables
- `PYTHONPATH=/app` (set automatically)

## Usage Examples

### Pull and Run from Registry
```bash
# Latest version
docker pull <username>/energy-efficiency-api:latest
docker run -p 8000:8000 <username>/energy-efficiency-api:latest

# Specific version
docker pull <username>/energy-efficiency-api:v1.0.0
docker run -p 8000:8000 <username>/energy-efficiency-api:v1.0.0
```

### Docker Compose
```bash
docker-compose up --build
```

### Production Deployment
```bash
# Run in detached mode with restart policy
docker run -d --restart=unless-stopped -p 8000:8000 --name ml-api <username>/energy-efficiency-api:latest
```

## Versioning Strategy

### Tag Format
- `latest` - Most recent build
- `v1.0.0` - Semantic versioning
- `v1.0.0-<commit-hash>` - Build-specific tags

### Recommended Tags
```bash
# Major release
docker tag ml-service:latest <username>/energy-efficiency-api:v1.0.0

# Minor update
docker tag ml-service:latest <username>/energy-efficiency-api:v1.1.0

# Patch
docker tag ml-service:latest <username>/energy-efficiency-api:v1.0.1
```

## Optimization Features

### .dockerignore
Excludes unnecessary files:
- Development environment (`env/`)
- Git history (`.git/`)
- Test files (`tests/`)
- Raw data (`data/raw/`)

### Multi-stage Build (Optional)
For even smaller images, consider multi-stage builds in production.
