# Docker setup for MLOPS project (mlops copy)

This repo contains a FastAPI backend (`main.py`) that loads a PyTorch model (`best_model.pth`) and exposes a `/predict` endpoint expecting a file upload. A simple static `frontend.html` is served by Nginx and proxied via an `nginx` reverse proxy.

Files added:
- `backend/Dockerfile` — builds Python image with FastAPI and Torch
- `backend/requirements.txt` — Python packages
- `Dockerfile.frontend` — simple Nginx image serving `frontend.html`
- `nginx/nginx.conf` — Nginx config that proxies `/api/` to backend and serves frontend
- `docker-compose.yml` — orchestrates `backend`, `frontend`, and `nginx`

How to run (Windows PowerShell):

1. Build and start containers:

```powershell
cd "C:\Users\natha\Desktop\Progra\Progra\9no-Semestre\MLOPS\mlops copy"
docker compose up --build -d
```

2. Check logs if something fails:

```powershell
docker compose logs -f
```

3. Test the services:
- Frontend: http://localhost
- Backend (direct): http://localhost:8000 (FastAPI docs at http://localhost:8000/docs)
- Nginx proxy: http://localhost -> serves frontend and proxies `/api/predict` to backend

Notes and caveats:
- Docker image with PyTorch can be large. Building the `backend` image may take several minutes and require sufficient RAM/disk space.
- If you prefer to serve frontend directly via Nginx container, `frontend` service can be removed and `nginx` can serve the static file directly (current configuration mounts `frontend.html` into nginx).
- Ensure Docker Desktop is running on Windows and WSL2 integration is enabled if required.

Stopping and removing:

```powershell
docker compose down --volumes --remove-orphans
```

If you want, I can:
- Replace the `backend` base image with an official `pytorch` image to avoid long pip installs (faster builds), or
- Create a multi-stage build to reduce image size, or
- Configure TLS and custom domain in `nginx`.
