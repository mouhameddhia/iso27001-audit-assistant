# Docker Setup Guide - ISO 27001 Audit Assistant

Complete Docker setup to run the entire application stack with one command.

---

## Prerequisites

✅ Docker Desktop installed (https://www.docker.com/products/docker-desktop)
✅ At least 8GB RAM recommended
✅ 20GB free disk space

---

## Quick Start (Windows)

### Automatic Setup (Recommended)

```powershell
# Run from project root directory
.\docker-setup.bat
```

This will:
1. Build all containers
2. Start all services
3. Pull LLM models (takes 5-10 minutes)
4. Show access URLs

### Manual Setup

```powershell
# From project root
docker compose up -d --build

# Wait 10 seconds
Start-Sleep -Seconds 10

# Pull models
docker exec iso27001_ollama ollama pull llama3:8b
docker exec iso27001_ollama ollama pull bge-m3
```

---

## Access the Application

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | React UI |
| Backend API | http://localhost:8000 | FastAPI |
| API Docs | http://localhost:8000/docs | Swagger |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector DB |

---

## First Login

When backend starts, it creates admin account:

```
Username: admin
Password: (check backend logs)
```

View password:
```powershell
docker compose logs backend | findstr "Bootstrapped"
```

---

## Stop Everything

```powershell
docker compose down
```

Stop and remove all data:
```powershell
docker compose down -v
```

---

## View Logs

All services:
```powershell
docker compose logs -f
```

Specific service:
```powershell
docker compose logs -f backend      # FastAPI
docker compose logs -f frontend     # React
docker compose logs -f postgres     # Database
docker compose logs -f qdrant       # Vector store
docker compose logs -f ollama       # LLM
```

---

## Service Details

### PostgreSQL (Database)
- Port: 5433
- User: audit_app
- Password: dev_local_only_change_me
- Database: iso27001_audit

Connect:
```powershell
docker exec -it iso27001_postgres psql -U audit_app -d iso27001_audit
```

### Qdrant (Vector Database)
- Port: 6333
- Dashboard: http://localhost:6333/dashboard
- Purpose: ISO 27001/27002 embeddings

Check collections:
```powershell
docker exec -it iso27001_qdrant curl http://localhost:6333/collections
```

### Ollama (LLM + Embeddings)
- Port: 11434
- Models: llama3:8b (~4.1GB), bge-m3 (~1.3GB)

Check models:
```powershell
docker exec iso27001_ollama ollama list
```

Test LLM:
```powershell
docker exec iso27001_ollama ollama run llama3:8b "Hello"
```

### FastAPI (Backend)
- Port: 8000
- Docs: http://localhost:8000/docs
- Hot reload: Enabled

Test health:
```powershell
curl http://localhost:8000/health
```

### React (Frontend)
- Port: 5173
- Build: Production-optimized
- Hot reload: Enabled

---

## Troubleshooting

### Port 8000 already in use

```powershell
$PID = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess
if ($PID) { Stop-Process -Id $PID -Force }

docker compose up -d
```

### Ollama model download stuck

```powershell
docker compose logs ollama

# Pull again
docker exec iso27001_ollama ollama pull llama3:8b
```

### Database connection refused

```powershell
docker compose ps
docker compose logs postgres
docker compose restart postgres
```

### Frontend not loading

```powershell
# Clear browser cache (Ctrl+Shift+R)

docker compose logs frontend

docker compose build --no-cache frontend
docker compose up -d frontend
```

---

## Rebuild Containers

```powershell
# Rebuild backend only
docker compose build --no-cache backend
docker compose up -d backend

# Rebuild frontend only
docker compose build --no-cache frontend
docker compose up -d frontend

# Rebuild all
docker compose build --no-cache
docker compose up -d
```

---

## Database Backup & Restore

### Backup:
```powershell
docker exec iso27001_postgres pg_dump -U audit_app iso27001_audit > backup.sql
```

### Restore:
```powershell
docker compose down -v
docker compose up -d postgres
Start-Sleep -Seconds 5
cat backup.sql | docker exec -i iso27001_postgres psql -U audit_app iso27001_audit
```

---

## Production Considerations

This Docker setup is for development only!

Production setup should include:
- Proper secrets management (not hardcoded)
- Managed database (AWS RDS, etc.)
- Docker registry (Docker Hub, ECR, etc.)
- HTTPS/TLS (Nginx reverse proxy)
- Monitoring (Prometheus, Grafana)
- Automated backups
- Kubernetes or Docker Swarm for scaling

---

## Environment Variables

All defined in `docker-compose.yml`:

```yaml
DATABASE_URL: postgresql+psycopg2://audit_app:dev_local_only_change_me@postgres:5432/iso27001_audit
JWT_SECRET_KEY: ${JWT_SECRET_KEY}   # set it in .env, never commit it
JWT_ALGORITHM: HS256
JWT_EXPIRE_MINUTES: 480
QDRANT_URL: http://qdrant:6333
QDRANT_COLLECTION: iso27001_kb
EMBEDDING_BASE_URL: http://ollama:11434
GENERATION_BASE_URL: http://ollama:11434
```

---

## Verification Checklist

After setup:

- [ ] Frontend loads: http://localhost:5173
- [ ] API docs: http://localhost:8000/docs
- [ ] Qdrant dashboard: http://localhost:6333/dashboard
- [ ] Can login with admin credentials
- [ ] Can create audit session
- [ ] No backend errors in logs

---

Ready to go! 🚀
