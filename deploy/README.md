# Cerebrum Blocks Deployment Configs

Containerized deployment configurations for cloud and edge environments. None of these are wired to an active deploy in this fork — see them as scaffolding for if/when you stand one up.

## Structure

```
deploy/
├── cloud/              # Docker configs for cloud deployment
│   ├── Dockerfile          # Main API container
│   └── Dockerfile.worker   # Celery worker container
├── gcp/                # Google Cloud Platform
│   └── cloudbuild.yaml     # Cloud Build + Cloud Run
├── edge/               # Edge/Jetson deployment
│   └── Dockerfile.jetson   # ARM64/CUDA optimized
└── README.md
```

## Quick Deploy

### Docker (local or any cloud)

```bash
# Local
docker compose up -d

# Build a specific target
docker build -f deploy/cloud/Dockerfile -t cerebrum:cloud .
docker run -p 8000:8000 -e PORT=8000 cerebrum:cloud
```

A prebuilt image is also published on each push to main:

```bash
docker pull ghcr.io/bopoadz-del/cerebrum-blocks:latest
```

### Google Cloud Platform

```bash
gcloud builds submit --config deploy/gcp/cloudbuild.yaml
gcloud run deploy cerebrum-api --image gcr.io/PROJECT/cerebrum-api:latest
```

### NVIDIA Jetson (Edge)

```bash
docker build -f deploy/edge/Dockerfile.jetson -t cerebrum:jetson .
docker run -p 8000:8000 --runtime nvidia cerebrum:jetson
```

## Required Secrets

Create a `.env` file in the repo root (see `.env.example` for the full template):

```bash
KIMI_API_KEY=...
GROQ_API_KEY=...

DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

## Health Check

`GET /v1/health` — JSON response. Hook any uptime monitor or container health probe to it.
