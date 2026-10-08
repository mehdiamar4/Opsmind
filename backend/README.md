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
uvicorn app.main:app --reload
```

The development server runs at <http://127.0.0.1:8000>.

## Check the endpoints

In another terminal, run:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/version
```

The responses are `{"status":"healthy"}` and `{"version":"0.1.0"}`.
