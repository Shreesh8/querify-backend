# Querify — AI Business Intelligence Backend

Production-grade FastAPI backend for an AI-powered analytics platform.

## What it does
- Upload CSV/Excel datasets
- Auto-generate analytics, charts, and statistics
- Ask natural language questions about your data
- Generate AI business insights using Gemini
- Forecast future trends using Prophet

## Tech Stack
FastAPI · PostgreSQL · Pandas · Prophet · Gemini AI · SQLAlchemy · Docker

## Architecture
- Layered service pattern (routes → services → models)
- Structured query execution (no eval/exec)
- Async SQLAlchemy with connection pooling
- JWT authentication
- Structured JSON logging

## Quick Start
```bash
cp .env.example .env   # fill in your keys
docker run -d --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:15
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs
