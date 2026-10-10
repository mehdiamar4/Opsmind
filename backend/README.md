# OpsMind Backend

Minimal FastAPI backend for OpsMind.

## Set up and run locally

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
set -a
source .env
set +a
uvicorn app.main:app --reload
```

The development server runs at <http://127.0.0.1:8000>.

## PostgreSQL and migrations

Install and start PostgreSQL locally, then create a database and matching local role. For example, with a local PostgreSQL installation:

```bash
createuser opsmind --createdb --pwprompt
createdb opsmind --owner=opsmind
```

For the example URL in `.env.example`, set the role password to `opsmind`, or change the URL to match the password you choose. Copy and source the file as shown above, then run the initial migration from the `backend/` directory:

```bash
alembic upgrade head
```

To see the current revision, use `alembic current`. To reverse the initial schema migration, use `alembic downgrade base`.

`DATABASE_URL` must be set in the environment for Alembic and for code that imports the database module. Keep real credentials in your local `.env`; do not commit that file.

## Chat endpoint

Set `OPENAI_API_KEY` to an API key for your OpenAI project and `OPENAI_MODEL` to a model available to that project. The example `.env` uses `gpt-6-luna`; replace its API key placeholder with your own key. The application does not load `.env` automatically, so source it in the shell using the commands above.

Apply the usage-event migration from the `backend/` directory:

```bash
alembic upgrade head
```

Send a request to `/chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How should I prepare for a team handoff?"}'
```

The response contains an answer and the model used. Raw input/output token counts, latency, success status, and a database-generated timestamp are recorded in `ai_usage_events`. Provider failures return a generic `502`; missing LLM configuration returns `503`.

## Check the endpoints

In another terminal, run:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/version
```

The responses are `{"status":"healthy"}` and `{"version":"0.1.0"}`.
