# Karna — local setup

Run this once. After that, `python main.py cli` is your daily driver.

## Prerequisites

- Python 3.11, 3.12, or 3.13 — all work for Sprint 1/2 deps. **Python 3.12 recommended** when we get to Sprint 3 (numpy/pandas/pandas-ta ecosystem is most settled there).
- Docker Desktop (running)
- ~10 GB free disk for the Llama model + Neo4j volume

## 1. Python environment

```powershell
# from the repo root
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The data-stack deps (numpy, pandas, yfinance, pandas-ta, FastAPI, etc.) live in **`requirements-data.txt`** and are only needed from Sprint 3 onwards. Install them later with `pip install -r requirements-data.txt`. Splitting them keeps the Sprint 1 install fast and avoids numpy build pain on Python 3.13.

## 2. Local infrastructure

```powershell
docker compose up -d
# pull the LLM model once (~5GB)
docker exec -it karna-ollama ollama pull llama3.1:8b
```

This brings up:
- Neo4j on `bolt://localhost:7687` (browser UI at http://localhost:7474)
- ChromaDB on `http://localhost:8000`
- Ollama on `http://localhost:11434`

## 3. Configuration

```powershell
copy .env.example .env
```

Then edit `.env`:
- Leave `NEO4J_PASSWORD=karnapassword` (matches docker-compose.yml).
- Get a Telegram bot token from [@BotFather](https://t.me/BotFather) — `/newbot`, follow prompts, paste the token into `TELEGRAM_BOT_TOKEN`. *(Skip if you only want CLI for now.)*
- Get your Telegram user id from [@userinfobot](https://t.me/userinfobot) and put it in `TELEGRAM_ALLOWED_USER_IDS` so randos can't hit your bot.

## 4. Health check

```powershell
python main.py health
```

You should see all four services OK.

## 5. Seed the concept graph

```powershell
python main.py seed-graph
```

Loads the trading concept tree into Neo4j. Idempotent — safe to re-run.

## 6. Talk to it

```powershell
# Terminal (works without Telegram):
python main.py cli

# Telegram (needs TELEGRAM_BOT_TOKEN):
python main.py telegram
```

## Running tests

```powershell
pytest tests/test_router_and_concepts.py tests/test_session_store.py
```

Those two suites don't need infra. (More tests land in Sprints 2-3.)

## Troubleshooting

**`Ollama unreachable`** — `docker ps` to check `karna-ollama` is running. If yes, the model may not be pulled yet: `docker exec -it karna-ollama ollama list`.

**`Neo4j unavailable`** — first start can take 30s. Wait, then re-run health check.

**`TELEGRAM_BOT_TOKEN not set`** — only blocks `python main.py telegram`. CLI works without it.

**Slow first reply** — Ollama loads the model into RAM on first call (~30s on 16GB). Subsequent calls are fast.
