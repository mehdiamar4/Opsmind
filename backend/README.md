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

## Check the endpoints

In another terminal, run:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/version
```

The responses are `{"status":"healthy"}` and `{"version":"0.1.0"}`.
