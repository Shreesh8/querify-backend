# Querify — Backend

The backend API for [Querify](https://querify.site), an AI-powered business intelligence platform. Handles dataset ingestion, natural language querying, forecasting, and AI-generated insights.

**Live APP:** [api.querify.site](https://api.querify.site)
**Frontend repo:** [querify-frontend](https://github.com/Shreesh8/querify-frontend)

## Tech Stack

- FastAPI (async Python)
- PostgreSQL (AWS RDS, async SQLAlchemy + Alembic migrations)
- Groq (`llama-3.3-70b-versatile`) for natural language querying and insights
- Prophet for time-series forecasting
- Firebase Admin SDK for auth (verifies Firebase ID tokens server-side)
- Pandas / NumPy / scikit-learn for data processing
- Hosted on AWS EC2, exposed over HTTPS via a Cloudflare Tunnel

## Architecture

Natural language queries don't get turned into executable code. Instead, the AI layer returns a **whitelisted JSON operation spec**, which a safe executor (`app/services/ai/operation_executor.py`) interprets against the dataset. This avoids `eval()`-ing model output entirely — a deliberate security tradeoff over flexibility.

The AI client sits behind a shared interface (`app/services/ai/groq_client.py`), so swapping providers (this project migrated from Gemini to Groq) doesn't require touching the rest of the query pipeline.

## Getting Started

### Prerequisites

- Python 3.11 (Docker) — production runs 3.8, see note below
- PostgreSQL instance
- Groq API key
- Firebase service account credentials

### Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### Environment variables

Create a `.env` file in the project root:

```
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
GROQ_API_KEY=your_groq_api_key
ADMIN_SECRET_KEY=your_admin_secret
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://querify.site
FIREBASE_CREDENTIALS_B64=base64_encoded_service_account_json
```

### Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t querify-backend .
docker run -p 8000:8000 --env-file .env querify-backend
```

## Deployment

Production runs on an AWS EC2 instance as a systemd service, exposed over HTTPS via a named Cloudflare Tunnel routing `api.querify.site` to `localhost:8000` — no public HTTP port, no mixed-content issues on the frontend.

## Project Structure

```
app/
  api/routes/       # FastAPI route handlers
  core/             # config, security, logging, exceptions
  db/               # SQLAlchemy models and session management
  services/
    ai/             # Groq client, prompt builder, query service, operation executor
    analytics/      # analytics engine
    datasets/       # upload and cleaning
    forecasting/     # Prophet-based forecast service
  schemas/          # Pydantic schemas
```

## Key Design Notes

- **NaN handling**: Python's `float('nan')` isn't valid JSON — all query responses pass through a sanitizer (`_sanitize_nans`) before serialization.
- **Cascading deletes**: dataset deletion explicitly cascades to related chat messages and forecasts before removing the dataset row.
- **Usage limits**: a free-tier query/forecast/dataset limit system backs the frontend's usage meter, with an admin endpoint for manual upgrades.
