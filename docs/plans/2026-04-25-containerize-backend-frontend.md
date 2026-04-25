# Containerize Backend and Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add production-optimized Docker containers for the FastAPI backend and the built (nginx-served) React frontend to the existing docker-compose, sharing the same `.env` file.

**Architecture:** The backend is a multi-stage Python image that installs dependencies then copies source; it reads all config from `.env` via `env_file`, with two overrides in compose for the service-name hostnames (`postgres`, `memgraph`). The frontend is a multi-stage Node build followed by an nginx stage that serves static files and reverse-proxies `/api` to the backend — no env vars needed at runtime. Both services depend on postgres and memgraph being healthy.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Node 20, Vite, nginx:alpine, Docker Compose v2.

---

### Task 1: Root `.dockerignore`

**Files:**
- Create: `.dockerignore`

No tests for this task — it is build-time hygiene only.

**Step 1: Create `.dockerignore` at the project root**

```
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
venv/
.venv/
*.egg-info/
dist/
build/

# Data (never bake into image)
data/

# Dev files
.env
.env.*
!.env.example
docs/
test_documents/
tests/

# Frontend (built separately)
frontend/

# Git
.git/
.gitignore
```

**Step 2: Commit**

```bash
git add .dockerignore
git commit -m "build: add root .dockerignore for backend image"
```

---

### Task 2: Backend Dockerfile

**Files:**
- Create: `Dockerfile`

No automated tests — verify manually in Task 5.

**Step 1: Create `Dockerfile` at project root**

```dockerfile
# ── build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps for compiled wheels (psycopg binary, igraph, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install -e .

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /app/src ./src

# Storage mount point (matches STORAGE_BASE_PATH=/data/documents in compose)
RUN mkdir -p /data/documents

EXPOSE 8000

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "build: add production backend Dockerfile"
```

---

### Task 3: Frontend Dockerfile and nginx config

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`

No automated tests — verify manually in Task 5.

**Step 1: Create `frontend/nginx.conf`**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # Reverse-proxy all API calls to the backend service
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    # SPA fallback — all other paths serve index.html
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**Step 2: Create `frontend/Dockerfile`**

```dockerfile
# ── build stage ──────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM nginx:alpine AS runtime

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/app.conf

EXPOSE 80
```

**Step 3: Commit**

```bash
git add frontend/Dockerfile frontend/nginx.conf
git commit -m "build: add production frontend Dockerfile with nginx"
```

---

### Task 4: Update docker-compose.yml and fix CORS

**Files:**
- Modify: `docker-compose.yml`
- Modify: `src/rag/api/main.py`

**Step 1: Update CORS origins in `src/rag/api/main.py`**

The dev origins (`localhost:5173`) must be replaced with the nginx container origin. In production the browser hits nginx on port 80, so the origin is `http://localhost` (or whatever host). Add both so local dev still works:

Old:
```python
allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
```

New:
```python
allow_origins=[
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
],
```

**Step 2: Add `backend` and `frontend` services to `docker-compose.yml`**

Append after the existing `memgraph` service:

```yaml
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-backend
    env_file:
      - .env
    environment:
      POSTGRES_URL: postgresql://${POSTGRES_USER:-rag}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-rag}
      MEMGRAPH_URL: bolt://memgraph:7687
      STORAGE_BASE_PATH: /data/documents
    volumes:
      - ./data/documents:/data/documents
    ports:
      - "${BACKEND_PORT:-8000}:8000"
    depends_on:
      postgres:
        condition: service_healthy
      memgraph:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: rag-frontend
    ports:
      - "${FRONTEND_PORT:-80}:80"
    depends_on:
      - backend
    restart: unless-stopped
```

**Step 3: Commit**

```bash
git add docker-compose.yml src/rag/api/main.py
git commit -m "feat: add backend and frontend services to docker-compose"
```

---

### Task 5: Smoke-test the full stack

No code changes — verify the containers build and run correctly.

**Step 1: Build both images**

```bash
docker compose build backend frontend
```
Expected: both builds complete with no errors.

**Step 2: Start everything**

```bash
docker compose up -d
```
Expected: all four containers reach `running` state. Postgres and memgraph must pass their healthchecks before backend starts.

**Step 3: Check backend health**

```bash
curl http://localhost:8000/api/health
```
Expected: `{"status":"ready"}`

**Step 4: Check frontend is served**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost/
```
Expected: `200`

**Step 5: Check frontend proxies API through nginx**

```bash
curl http://localhost/api/health
```
Expected: `{"status":"ready"}` (same response, routed through nginx)

**Step 6: Commit nothing** — if all checks pass, the feature is complete.

---

### Task 6: Document the new ports in `.env.example`

**Files:**
- Modify: `.env.example`

**Step 1: Add port override vars to `.env.example`**

Add after the `MEMGRAPH_LAB_PORT` line:

```
# Application ports (optional — defaults are fine)
BACKEND_PORT=8000
FRONTEND_PORT=80
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: document BACKEND_PORT and FRONTEND_PORT in .env.example"
```
