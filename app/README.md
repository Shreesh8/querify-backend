# Insight Engine — Backend

AI-powered Business Intelligence platform  
**Stack:** FastAPI · PostgreSQL · Pandas · Prophet · Gemini API

---

## Local Development Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Node (for frontend, separate repo)

### 1. Clone and navigate
```bash
cd backend
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, GEMINI_API_KEY, SECRET_KEY
```

### 5. Start PostgreSQL (Docker)
```bash
docker run -d \
  --name insight-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=insight_engine \
  -p 5432:5432 \
  postgres:15
```

### 6. Run the server
```bash
# Development (auto-reload + auto table creation)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or via main.py
python -m app.main
```

### 7. Verify
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/datasets/upload` | Upload CSV/Excel |
| GET  | `/api/v1/datasets/` | List user's datasets |
| GET  | `/api/v1/datasets/{id}` | Dataset metadata |
| GET  | `/api/v1/datasets/{id}/preview` | Preview + column info |
| DELETE | `/api/v1/datasets/{id}` | Delete dataset |
| GET  | `/api/v1/analytics/{dataset_id}` | Full analytics |
| POST | `/api/v1/chat/query` | NL query |
| GET  | `/api/v1/insights/{dataset_id}` | AI insights |
| POST | `/api/v1/forecast/generate` | Prophet forecast |
| GET  | `/health` | Health check |

---

## Database Migrations (Production)

```bash
# Init alembic (first time only)
alembic init alembic

# Generate migration
alembic revision --autogenerate -m "initial"

# Apply
alembic upgrade head
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Deployment (Render)

1. Connect GitHub repo to Render
2. Set environment variables in Render dashboard
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| Structured query specs (not eval) | LLM generates whitelisted JSON ops, not executable code |
| Async SQLAlchemy | Non-blocking DB — handles concurrent users |
| Thread pool for Pandas/Prophet | CPU-bound work offloaded from async event loop |
| Service layer | Business logic decoupled from HTTP layer, fully testable |
| Pydantic everywhere | Type safety + automatic API docs |
| structlog | JSON logs for production observability |
