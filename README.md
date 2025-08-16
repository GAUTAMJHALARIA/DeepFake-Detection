## Deepfake Detection – Production Stack

- API: FastAPI
- Serving: TensorFlow Serving (SavedModel)
- Queue: Redis + Celery (later)
- Storage: MinIO (later)
- Proxy/LB: Nginx
- Orchestration: Docker Compose (dev), Kubernetes + Helm (prod)
- Observability: Prometheus + Grafana, ELK
- Security: OAuth2/JWT, TLS via Let’s Encrypt
